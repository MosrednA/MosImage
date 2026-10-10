import { Dispatch, MutableRefObject, SetStateAction, useEffect, useRef, useState } from "react";
import { DocumentRecord, LayerRecord, getDocument, saveLayers, uploadLayerAsset } from "../api";
import { canvasBlob } from "../layers/bitmaps";
import { initialPlacement } from "../layers/geometry";
import { MAX_LAYERS, duplicateLayer, insertResult, layerName, moveLayer, newLayerId, patchLayer } from "../layers/stack";

const ACCEPTED = new Set(["image/png", "image/jpeg", "image/webp"]);
const NUDGE_SETTLE = 450;

interface Draft { id: string; patch: Partial<LayerRecord> }

/**
 * The layer stack of the open document. Changes show at once and are saved in
 * order as undoable revisions; `documentRef` always holds the newest revision so
 * other requests can build on it before React re-renders.
 */
export function useLayers(
  document: DocumentRecord | null,
  setDocument: Dispatch<SetStateAction<DocumentRecord | null>>,
  documentRef: MutableRefObject<DocumentRecord | null>,
  onError: (message: string) => void,
) {
  const [activeId, setActiveId] = useState<string | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [uploading, setUploading] = useState(0);
  const queue = useRef<Promise<void>>(Promise.resolve());
  const pending = useRef(0);
  const failed = useRef(false);
  const nudgeTimer = useRef<number | null>(null);

  const committed = document?.layers ?? [];
  const layers = draft ? patchLayer(committed, draft.id, draft.patch) : committed;
  const active = layers.find(layer => layer.id === activeId) ?? null;

  useEffect(() => { setActiveId(null); setDraft(null); }, [document?.id]);
  // Undo can take the selected layer away.
  useEffect(() => { if (activeId && !committed.some(layer => layer.id === activeId)) setActiveId(null); }, [committed, activeId]);

  function show(next: DocumentRecord) {
    documentRef.current = next;
    setDocument(next);
  }

  function commit(next: LayerRecord[], label: string) {
    const start = documentRef.current;
    if (!start) return;
    setDraft(null);
    show({ ...start, layers: next });
    pending.current += 1;
    queue.current = queue.current.then(async () => {
      try {
        const latest = documentRef.current;
        if (failed.current || !latest || latest.id !== start.id) return;
        const saved = await saveLayers(start.id, latest.current_revision_id ?? 0, next, label);
        const current = documentRef.current;
        if (current?.id !== start.id) return;
        // Keep showing newer changes that are still on their way.
        show({ ...saved, name: current.name, layers: pending.current > 1 ? current.layers : saved.layers });
      } catch (err) {
        failed.current = true;
        onError(message(err));
        try {
          const fresh = await getDocument(start.id);
          if (documentRef.current?.id === start.id) show({ ...fresh, name: documentRef.current.name });
        } catch { /* The engine went away; the error above says enough. */ }
      } finally {
        pending.current -= 1;
        if (!pending.current) failed.current = false;
      }
    });
  }

  /** Resolves once every change so far is saved. */
  function settled(): Promise<void> {
    flushNudge();
    return queue.current;
  }

  function update(id: string, patch: Partial<LayerRecord>, label: string) {
    commit(patchLayer(currentLayers(), id, patch), label);
  }

  function currentLayers(): LayerRecord[] { return documentRef.current?.layers ?? []; }

  async function addImages(files: File[]): Promise<void> {
    const images = files.filter(file => ACCEPTED.has(file.type));
    if (images.length < files.length) onError("Layers can be PNG, JPG or WebP images.");
    for (const file of images) await addImage(file, file.name);
  }

  /** Uploads an image as a new layer; centred and fitted unless a placement is given. */
  async function addImage(image: Blob, filename: string, placement?: { x: number; y: number }): Promise<LayerRecord | null> {
    const start = documentRef.current;
    if (!start) return null;
    if (currentLayers().length >= MAX_LAYERS) { onError(`A document can have up to ${MAX_LAYERS} layers.`); return null; }
    setUploading(count => count + 1);
    try {
      const asset = await uploadLayerAsset(start.id, image, "image", filename.replace(/\.[^.]+$/, "") + ".png");
      if (documentRef.current?.id !== start.id) return null;
      const fitted = initialPlacement(asset, { width: start.canvas_width, height: start.canvas_height });
      const layer: LayerRecord = {
        id: newLayerId(), name: layerName(filename, currentLayers().length + 1), image_path: asset.path, image_url: asset.url,
        mask_path: null, mask_url: null, mask_enabled: true, width: asset.width, height: asset.height, opacity: 1, visible: true,
        ...fitted, ...(placement ? { ...placement, scale_x: 1, scale_y: 1 } : {}),
      };
      commit([...currentLayers(), layer], "Add layer");
      setActiveId(layer.id);
      return layer;
    } catch (err) {
      onError(message(err));
      return null;
    } finally {
      setUploading(count => count - 1);
    }
  }

  /**
   * Adds a full generation or selected AI patch at its place and scale.
   * With `replaces`, it goes just above that layer, which is hidden.
   */
  async function addResult(image: Blob, mask: Blob | null, name: string, placement: { x: number; y: number; scale_x?: number; scale_y?: number; rotation?: number; opacity?: number; mask_enabled?: boolean }, replaces: string | null, label?: string): Promise<LayerRecord | null> {
    const start = documentRef.current;
    if (!start) return null;
    if (currentLayers().length >= MAX_LAYERS) throw new Error(`A document can have up to ${MAX_LAYERS} layers. Delete one to apply this result.`);
    setUploading(count => count + 1);
    try {
      const [asset, maskAsset] = await Promise.all([
        uploadLayerAsset(start.id, image, "image", "result.png"),
        mask ? uploadLayerAsset(start.id, mask, "mask", "mask.png") : Promise.resolve(null),
      ]);
      if (documentRef.current?.id !== start.id) return null;
      const layer: LayerRecord = {
        id: newLayerId(), name: name.slice(0, 80), image_path: asset.path, image_url: asset.url,
        mask_path: maskAsset?.path ?? null, mask_url: maskAsset?.url ?? null, mask_enabled: placement.mask_enabled ?? true, width: asset.width, height: asset.height,
        x: placement.x, y: placement.y, scale_x: placement.scale_x ?? 1, scale_y: placement.scale_y ?? 1, rotation: placement.rotation ?? 0, opacity: placement.opacity ?? 1, visible: true,
      };
      commit(insertResult(currentLayers(), layer, replaces), label ?? (replaces ? "Blend layer in" : "Add AI result"));
      await settled();
      // Keep the proposal available when saving failed and restored the stack.
      if (!currentLayers().some(item => item.id === layer.id)) return null;
      setActiveId(layer.id);
      return layer;
    } finally {
      setUploading(count => count - 1);
    }
  }

  /** Stores a new mask (coverage in alpha) for a layer and switches it on. */
  async function setMask(id: string, mask: HTMLCanvasElement | Blob | null, label: string) {
    const start = documentRef.current;
    if (!start) return;
    try {
      if (!mask) { update(id, { mask_path: null, mask_url: null, mask_enabled: true }, label); return; }
      const blob = mask instanceof Blob ? mask : await canvasBlob(mask);
      const asset = await uploadLayerAsset(start.id, blob, "mask", "mask.png");
      if (documentRef.current?.id !== start.id) return;
      update(id, { mask_path: asset.path, mask_url: asset.url, mask_enabled: true }, label);
    } catch (err) {
      onError(message(err));
    }
  }

  function remove(id: string) {
    commit(currentLayers().filter(layer => layer.id !== id), "Delete layer");
    if (activeId === id) setActiveId(null);
  }

  function duplicate(id: string) {
    if (currentLayers().length >= MAX_LAYERS) { onError(`A document can have up to ${MAX_LAYERS} layers.`); return; }
    const result = duplicateLayer(currentLayers(), id);
    if (!result) return;
    commit(result.layers, "Duplicate layer");
    setActiveId(result.copy.id);
  }

  function reorder(id: string, index: number) {
    const next = moveLayer(currentLayers(), id, index);
    if (next.some((layer, position) => layer.id !== currentLayers()[position]?.id)) commit(next, "Reorder layers");
  }

  /** Shows a change without saving it, for drags and sliders. */
  function preview(id: string, patch: Partial<LayerRecord>) { setDraft({ id, patch }); }
  function cancelPreview() { setDraft(null); }

  /** Arrow-key moves are saved once the keys rest, as one undo step. */
  function nudge(id: string, dx: number, dy: number) {
    const layer = layers.find(item => item.id === id);
    if (!layer) return;
    setDraft({ id, patch: { ...(draft?.id === id ? draft.patch : {}), x: layer.x + dx, y: layer.y + dy } });
    if (nudgeTimer.current !== null) window.clearTimeout(nudgeTimer.current);
    nudgeTimer.current = window.setTimeout(flushNudge, NUDGE_SETTLE);
  }
  const draftRef = useRef(draft);
  draftRef.current = draft;
  function flushNudge() {
    if (nudgeTimer.current === null) return;
    window.clearTimeout(nudgeTimer.current);
    nudgeTimer.current = null;
    const pendingDraft = draftRef.current;
    if (pendingDraft) update(pendingDraft.id, pendingDraft.patch, "Move layer");
  }
  useEffect(() => () => { if (nudgeTimer.current !== null) window.clearTimeout(nudgeTimer.current); }, []);

  return {
    layers, active, activeId, select: setActiveId, uploading,
    commit, update, preview, cancelPreview, nudge, settled,
    addImages, addImage, addResult, setMask, remove, duplicate, reorder,
  };
}

export type LayersState = ReturnType<typeof useLayers>;

function message(error: unknown): string { return error instanceof Error ? error.message : "The layer could not be saved."; }
