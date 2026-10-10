import { Dispatch, MutableRefObject, SetStateAction, useEffect, useRef, useState } from "react";
import { DocumentRecord, LayerRecord, getDocument, saveLayers, uploadLayerAsset } from "../api";
import { canvasBlob, mergeLayers } from "../layers/bitmaps";
import { initialPlacement } from "../layers/geometry";
import {
  MAX_LAYERS, MergeKind, duplicateLayer, insertResult, layerName, mergeBounds, mergePlan, moveLayer, newLayerId, patchLayer, rangeIds,
  removeLayers, splitVariants, stepVariant, withMerged,
} from "../layers/stack";
import { flatLayers } from "./usePicture";

const ACCEPTED = new Set(["image/png", "image/jpeg", "image/webp"]);
const NUDGE_SETTLE = 450;
/** The longest side the engine keeps for a layer image. */
const MAX_LAYER_SIDE = 8192;
const MERGE_LABELS: Record<MergeKind, string> = { down: "Merge down", selected: "Merge layers", visible: "Merge visible", flatten: "Flatten image" };

interface Draft { id: string; patch: Partial<LayerRecord> }
/** The selected layers; `active` is the one the tools and settings work on. */
interface Selection { active: string | null; ids: string[] }
const NOTHING: Selection = { active: null, ids: [] };

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
  const [selection, setSelection] = useState<Selection>(NOTHING);
  // Where Shift-click selects from, as in Photoshop.
  const anchor = useRef<string | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [uploading, setUploading] = useState(0);
  const queue = useRef<Promise<void>>(Promise.resolve());
  const pending = useRef(0);
  const failed = useRef(false);
  const nudgeTimer = useRef<number | null>(null);

  const committed = document?.layers ?? [];
  const layers = draft ? patchLayer(committed, draft.id, draft.patch) : committed;
  const activeId = selection.active;
  const active = layers.find(layer => layer.id === activeId) ?? null;

  useEffect(() => { setSelection(NOTHING); setDraft(null); }, [document?.id]);
  // Undo, merges and deletes can take selected layers away.
  useEffect(() => {
    const ids = selection.ids.filter(id => committed.some(layer => layer.id === id));
    if (ids.length === selection.ids.length) return;
    setSelection({ active: selection.active && ids.includes(selection.active) ? selection.active : ids[ids.length - 1] ?? null, ids });
  }, [committed, selection]);

  /** Selects one layer, or none. */
  function select(id: string | null) {
    setSelection(id ? { active: id, ids: [id] } : NOTHING);
    anchor.current = id;
  }
  /** Ctrl-click: adds a layer to the selection, or takes it out. */
  function toggle(id: string) {
    anchor.current = id;
    setSelection(current => {
      if (!current.ids.includes(id)) return { active: id, ids: [...current.ids, id] };
      const ids = current.ids.filter(item => item !== id);
      return { active: current.active === id ? ids[ids.length - 1] ?? null : current.active, ids };
    });
  }
  /** Shift-click: the layers from the last one clicked to this one. */
  function selectRange(id: string) {
    const ids = rangeIds(currentLayers(), anchor.current ?? selection.active ?? id, id);
    setSelection({ active: id, ids: ids.length ? ids : [id] });
  }

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
        mask_path: null, mask_url: null, mask_enabled: true, variant_paths: [], variant_urls: [], width: asset.width, height: asset.height, opacity: 1, visible: true, original_id: null,
        ...fitted, ...(placement ? { ...placement, scale_x: 1, scale_y: 1 } : {}),
      };
      commit([...currentLayers(), layer], "Add layer");
      select(layer.id);
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
   * Several images become variants of the one layer, showing `shown`, like
   * Photoshop's generative layers. With `replaces`, it goes just above that
   * layer, which is hidden. `separate` gives each image a layer of its own
   * instead, in the same undo step.
   */
  async function addResult(images: Blob[], mask: Blob | null, name: string, placement: { x: number; y: number; scale_x?: number; scale_y?: number; rotation?: number; opacity?: number; mask_enabled?: boolean }, replaces: string | null, label?: string, shown = 0, separate = false): Promise<LayerRecord | null> {
    const start = documentRef.current;
    if (!start) return null;
    const needed = separate ? images.length : 1;
    if (currentLayers().length + needed > MAX_LAYERS) throw new Error(`A document can have up to ${MAX_LAYERS} layers. Delete ${needed > 1 ? "some" : "one"} to apply this result.`);
    setUploading(count => count + 1);
    try {
      const [assets, maskAsset] = await Promise.all([
        Promise.all(images.map(image => uploadLayerAsset(start.id, image, "image", "result.png"))),
        mask ? uploadLayerAsset(start.id, mask, "mask", "mask.png") : Promise.resolve(null),
      ]);
      if (documentRef.current?.id !== start.id) return null;
      const asset = assets[shown] ?? assets[0];
      // Variants share the mask and transform, so only same-sized ones can stay.
      const variants = assets.filter(item => item.width === asset.width && item.height === asset.height);
      const layer: LayerRecord = {
        id: newLayerId(), name: name.slice(0, 80), image_path: asset.path, image_url: asset.url,
        variant_paths: variants.length > 1 ? variants.map(item => item.path) : [], variant_urls: variants.length > 1 ? variants.map(item => item.url) : [],
        mask_path: maskAsset?.path ?? null, mask_url: maskAsset?.url ?? null, mask_enabled: placement.mask_enabled ?? true, width: asset.width, height: asset.height,
        x: placement.x, y: placement.y, scale_x: placement.scale_x ?? 1, scale_y: placement.scale_y ?? 1, rotation: placement.rotation ?? 0, opacity: placement.opacity ?? 1, visible: true,
        original_id: null,
      };
      const inserted = insertResult(currentLayers(), layer, replaces);
      const split = separate ? splitVariants(inserted, layer.id) : null;
      commit(split?.layers ?? inserted, split ? "Add AI results" : label ?? (replaces ? "Blend layer in" : "Add AI result"));
      await settled();
      // Keep the proposal available when saving failed and restored the stack.
      if (!currentLayers().some(item => item.id === layer.id)) return null;
      select(layer.id);
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

  /** Deletes layers; the originals that deleted AI layers replaced show again. */
  function remove(ids: string[]) {
    if (!ids.length) return;
    commit(removeLayers(currentLayers(), ids), ids.length > 1 ? "Delete layers" : "Delete layer");
  }

  function setVisible(ids: string[], visible: boolean) {
    const gone = new Set(ids);
    const next = currentLayers().map(layer => gone.has(layer.id) ? { ...layer, visible } : layer);
    commit(next, (visible ? "Show " : "Hide ") + (ids.length > 1 ? "layers" : "layer"));
  }

  function duplicate(id: string) {
    if (currentLayers().length >= MAX_LAYERS) { onError(`A document can have up to ${MAX_LAYERS} layers.`); return; }
    const result = duplicateLayer(currentLayers(), id);
    if (!result) return;
    commit(result.layers, "Duplicate layer");
    select(result.copy.id);
  }

  /** What a merge would do with the stack and selection as they are now; null when it has nothing to merge. */
  function mergeable(kind: MergeKind) {
    const current = documentRef.current;
    if (!current) return null;
    const ids = kind === "down" ? (selection.active ? [selection.active] : []) : selection.ids;
    return mergePlan(current.layers, kind, ids, { width: current.canvas_width, height: current.canvas_height });
  }

  /**
   * Merges layers into one new layer, as one undo step. The merged layer has
   * no mask and shows exactly what its parts showed. Returns it with how many
   * hidden layers went too, or null when nothing was merged.
   */
  async function merge(kind: MergeKind): Promise<{ id: string; discarded: number } | null> {
    const start = documentRef.current;
    const plan = mergeable(kind);
    if (!start || !plan) return null;
    const size = { width: start.canvas_width, height: start.canvas_height };
    setUploading(count => count + 1);
    try {
      const box = mergeBounds(plan.parts, size, kind === "flatten");
      const image = await mergeLayers(await flatLayers(plan.parts), box, MAX_LAYER_SIDE);
      const asset = await uploadLayerAsset(start.id, image, "image", "merged.png");
      if (documentRef.current?.id !== start.id) return null;
      const merged: LayerRecord = {
        id: newLayerId(), name: plan.name, image_path: asset.path, image_url: asset.url, mask_path: null, mask_url: null, mask_enabled: true,
        variant_paths: [], variant_urls: [], width: asset.width, height: asset.height, x: box.x + box.width / 2, y: box.y + box.height / 2,
        // A merge wider than the engine keeps is stored smaller and scaled back up.
        scale_x: box.width / asset.width, scale_y: box.height / asset.height, rotation: 0, opacity: 1, visible: true, original_id: null,
      };
      commit(withMerged(currentLayers(), plan, merged), MERGE_LABELS[kind]);
      select(merged.id);
      return { id: merged.id, discarded: plan.removed.length - plan.parts.length };
    } finally {
      setUploading(count => count - 1);
    }
  }

  /** Shows the next or previous variant of a generative layer. */
  function switchVariant(id: string, step: number) {
    const layer = currentLayers().find(item => item.id === id);
    const patch = layer && stepVariant(layer, step);
    if (patch) update(id, patch, "Switch variant");
  }

  /** Gives every variant a layer of its own. */
  function separateVariants(id: string) {
    const result = splitVariants(currentLayers(), id);
    if (!result) return;
    if (currentLayers().length + result.added > MAX_LAYERS) { onError(`A document can have up to ${MAX_LAYERS} layers. Delete some before splitting the variants.`); return; }
    commit(result.layers, "Variants to layers");
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
    layers, active, activeId, selectedIds: selection.ids, select, toggle, selectRange, uploading,
    commit, update, preview, cancelPreview, nudge, settled,
    addImages, addImage, addResult, setMask, remove, setVisible, duplicate, reorder, switchVariant, separateVariants, mergeable, merge,
  };
}

export type LayersState = ReturnType<typeof useLayers>;

function message(error: unknown): string { return error instanceof Error ? error.message : "The layer could not be saved."; }
