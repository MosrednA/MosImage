import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import {
  DocumentRecord, Expansion, GenerationRecord, JobResponse, JobStatus, LayerRecord, ModelInfo, RevisionRecord, SelectionPoint, SystemInfo,
  acceptAlternative, cancelJob, createDocument, fileUrl, generateImage, getDocument, getGeneration, getSystemInfo,
  importGeneration, inpaintImage, listDocuments, listHistory, listModels, outpaintImage, proposeAlternatives, redoDocument,
  selectObject, undoDocument, updateDocument,
} from "./api";
import { EditorCanvas } from "./canvas/EditorCanvas";
import { CanvasTool, toolForKey } from "./canvas/tools";
import { ImageSize } from "./canvas/coordinates";
import { ExpansionCanvas } from "./canvas/ExpansionCanvas";
import { validExpansion } from "./canvas/expansion";
import { GENERATION_PRESETS, isKleinModel } from "./generation/presets";
import { Operation, operationHint, taskFor } from "./generation/operation";
import { SettingsControls, defaultSettings, settingsSummary } from "./components/EditorSettings";
import { Icon } from "./components/Icon";
import { ControlDialog } from "./components/ControlDialog";
import { LibraryDrawer } from "./components/LibraryDrawer";
import { TopBar } from "./components/TopBar";
import { ToolRail } from "./components/ToolRail";
import { CommandCard } from "./components/CommandCard";
import { ProposalTray } from "./components/ProposalTray";
import { JobOverlay } from "./components/JobOverlay";
import { ExpansionDialog } from "./components/ExpansionDialog";
import { WelcomeCanvas } from "./components/WelcomeCanvas";
import { ShortcutsDialog } from "./components/ShortcutsDialog";
import { ReferencePanel } from "./components/ReferencePanel";
import { LayersPanel } from "./components/LayersPanel";
import { SidePanel, SideTab } from "./components/SidePanel";
import { LayerStage } from "./layers/LayerStage";
import { cutSelection, flatten, invertMask, loadImage, loadMask, solidMask } from "./layers/bitmaps";
import { initialPlacement } from "./layers/geometry";
import { useLayers } from "./hooks/useLayers";
import { CancellationRequest, isActiveJob, useJobPolling } from "./hooks/useJobPolling";
import { MAX_ACTIVE_REFERENCES, useReferences } from "./hooks/useReferences";
import { useEngineWarmUp } from "./hooks/useEngineWarmUp";

interface JobContext {
  jobId: string;
  documentId: number | null;
  revisionId: number | null;
  operation: "inpaint" | "outpaint" | null;
  /** An object selection on this layer cuts it out, instead of selecting on the background. */
  layerId: string | null;
  /** Expand adds canvas on the left and top; layers move along with the content. */
  layerOffset: { x: number; y: number };
}
interface JobExtras { layerId?: string; layerOffset?: { x: number; y: number } }
const initialExpansion: Expansion = { left: 128, top: 0, right: 128, bottom: 0 };
const PANEL_COLLAPSED_KEY = "mosimage.sidePanelCollapsed";

/** Python and model imports take a while on a cold start. */
const ENGINE_START_PATIENCE = 90_000;
const ENGINE_RETRY_INTERVAL = 1000;

export default function App() {
  const [settings, setSettings] = useState(defaultSettings);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [history, setHistory] = useState<GenerationRecord[]>([]);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [document, setDocument] = useState<DocumentRecord | null>(null);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [online, setOnline] = useState(false);
  const [connecting, setConnecting] = useState(true);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<File | null>(null);
  const [size, setSize] = useState<ImageSize | null>(null);
  const [mask, setMask] = useState<Blob | null>(null);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [points, setPoints] = useState<SelectionPoint[]>([]);
  const [clearToken, setClearToken] = useState(0);
  const [tool, setTool] = useState<CanvasTool>("object");
  const [operation, setOperation] = useState<Operation>("generate");
  const [expansion, setExpansion] = useState(initialExpansion);
  const [alternatives, setAlternatives] = useState<GenerationRecord[]>([]);
  const [proposal, setProposal] = useState<RevisionRecord | null>(null);
  const [selected, setSelected] = useState(0);
  const [compare, setCompare] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [dropActive, setDropActive] = useState(false);
  const [panel, setPanel] = useState<"settings" | "expansion" | "shortcuts" | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [panelCollapsed, setPanelCollapsed] = useState(() => { try { return localStorage.getItem(PANEL_COLLAPSED_KEY) === "1"; } catch { return false; } });
  const [sideTab, setSideTab] = useState<SideTab>("layers");
  const [layerPoints, setLayerPoints] = useState<SelectionPoint[]>([]);
  const refs = useReferences(document?.id ?? null, setError);
  // Layer saves move the revision on before React re-renders; requests read it here.
  const documentRef = useRef<DocumentRecord | null>(document);
  documentRef.current = document;
  const layers = useLayers(document, setDocument, documentRef, setError);
  const layerFiles = useRef(new Map<string, File>());
  const upload = useRef<HTMLInputElement>(null);
  const context = useRef<JobContext | null>(null);
  const cancellation = useRef<CancellationRequest | null>(null);
  const transaction = useRef(false);

  const jobActive = isActiveJob(job);
  const cancelling = job?.status === "cancelling";
  const busy = pending || jobActive;
  const reviewing = alternatives.length > 0;
  const locked = busy || reviewing;
  const chosen = alternatives[selected];
  const proposalUrls = alternatives.map((item, index) => fileUrl(proposal?.alternatives[index]?.output_image_url ?? item.output_image_url));
  const previewUrl = chosen && !compare ? proposalUrls[selected] : null;
  const task = taskFor(operation);
  // References apply to Replace edits only; Remove and Expand ignore them.
  const references = operation === "replace" && source ? refs.active : [];
  const referenceNeedsQuality = references.length > 0 && !isKleinModel(settings.model_name);
  const activeLayer = operation === "expand" ? null : layers.active;
  const cudaBlocked = !!system && (!system.torch_available || (system.preferred_device === "cuda" && !system.cuda_available && !system.allow_cpu_generation));

  useEffect(() => setLayerPoints([]), [layers.activeId]);

  useEffect(() => {
    const url = source ? URL.createObjectURL(source) : null;
    setSourceUrl(url);
    return () => { if (url) URL.revokeObjectURL(url); };
  }, [source]);

  async function refresh() {
    const [nextHistory, nextDocuments] = await Promise.all([listHistory(), listDocuments()]);
    setHistory(nextHistory); setDocuments(nextDocuments);
  }
  /** Connects to the engine, retrying for `patience` ms while it starts up. */
  async function connect(patience = 0) {
    const deadline = Date.now() + patience;
    setConnecting(true);
    for (;;) {
      try {
        const [nextModels, nextSystem] = await Promise.all([listModels(), getSystemInfo(), refresh()]);
        setModels(nextModels); setSystem(nextSystem); setOnline(true); setError(null);
        break;
      } catch {
        if (Date.now() < deadline) { await new Promise(resolve => window.setTimeout(resolve, ENGINE_RETRY_INTERVAL)); continue; }
        setOnline(false); setError("The local engine is unavailable. Reopen the MosImage app or run start.cmd, then reconnect.");
        break;
      }
    }
    setConnecting(false);
  }
  // The desktop app starts the engine alongside the window, so give it time to come up.
  useEffect(() => { void connect(ENGINE_START_PATIENCE); }, []);
  const warmUp = useEngineWarmUp(online, setModels);

  // Debounced autosave of the document name.
  useEffect(() => {
    if (!document || !document.name.trim()) return;
    const { id, name } = document;
    const timer = window.setTimeout(() => {
      void updateDocument(id, name).then(saved => {
        setDocument(current => current?.id === id && current.name === name ? { ...current, updated_at: saved.updated_at } : current);
        setDocuments(current => current.map(item => item.id === id ? { ...item, name: saved.name } : item));
      }).catch(err => setError(message(err)));
    }, 650);
    return () => window.clearTimeout(timer);
  }, [document?.id, document?.name]);

  useEffect(() => {
    const paste = (event: ClipboardEvent) => {
      if (locked) return;
      const file = Array.from(event.clipboardData?.files ?? []).find(item => item.type.startsWith("image/"));
      if (!file) return;
      event.preventDefault();
      // With a picture open, a pasted image goes on top of it as a layer.
      if (document && source) void addLayerImages([file]);
      else void perform(() => openFile(file));
    };
    const keys = (event: KeyboardEvent) => {
      const typing = event.target instanceof HTMLElement && !!event.target.closest("input, textarea, select, [contenteditable], dialog");
      if (typing) return;
      const key = event.key.toLowerCase();
      const ctrl = event.ctrlKey || event.metaKey;
      if (event.key === "?") { setPanel("shortcuts"); return; }
      if (reviewing && !busy) {
        if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
          event.preventDefault();
          const step = event.key === "ArrowRight" ? 1 : -1;
          setSelected(index => (index + step + alternatives.length) % alternatives.length); setCompare(false);
        }
        if (ctrl && event.key === "Enter") { event.preventDefault(); void apply(); }
        return;
      }
      if (locked) return;
      if (activeLayer && !ctrl && !event.altKey) {
        const arrows: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
        const step = event.shiftKey ? 10 : 1;
        if (arrows[event.key]) { event.preventDefault(); layers.nudge(activeLayer.id, arrows[event.key][0] * step, arrows[event.key][1] * step); return; }
        if (event.key === "Backspace" && tool === "object" && layerPoints.length) { event.preventDefault(); void removeLayerPoint(layerPoints.length - 1); return; }
        if (event.key === "Delete" || event.key === "Backspace") { event.preventDefault(); layers.remove(activeLayer.id); return; }
        if (event.key === "Escape") { if (layerPoints.length) setLayerPoints([]); else layers.select(null); return; }
      }
      if (ctrl && key === "j" && mask && source) { event.preventDefault(); void copySelectionToLayer(); return; }
      if (ctrl && key === "o") { event.preventDefault(); upload.current?.click(); }
      if (ctrl && key === "z" && document) {
        event.preventDefault();
        if (event.shiftKey ? document.can_redo : document.can_undo) void perform(() => travel(event.shiftKey ? "redo" : "undo"));
      }
      if (!ctrl && !event.altKey && source) {
        if (key === "c") { chooseExpansion(); return; }
        const next = toolForKey(key);
        if (next) chooseTool(next);
      }
    };
    window.addEventListener("paste", paste); window.addEventListener("keydown", keys);
    return () => { window.removeEventListener("paste", paste); window.removeEventListener("keydown", keys); };
  }, [locked, busy, reviewing, alternatives.length, selected, proposal, document, source, models, settings.model_name, operation, activeLayer, layerPoints, tool, mask]);

  useJobPolling({
    job, setJob, cancellation,
    onError: setError,
    onDisconnected: () => setOnline(false),
    onCompleted: async (next, isDisposed) => {
      const submitted = context.current?.jobId === next.id ? context.current : null;
      if (next.mask_image_url) {
        if (submitted?.layerId) await cutOutLayer(submitted.layerId, fileUrl(next.mask_image_url));
        else setMaskUrl(fileUrl(next.mask_image_url));
      }
      if (next.generation_ids.length) {
        const generated = await Promise.all(next.generation_ids.map(getGeneration));
        const revision = submitted?.documentId && submitted.operation
          ? await proposeAlternatives(submitted.documentId, next.generation_ids, submitted.operation, submitted.revisionId ?? undefined, submitted.layerOffset) : null;
        if (isDisposed()) return;
        setAlternatives(generated); setProposal(revision); setSelected(0); setCompare(false);
        await refresh();
      }
      setModels(await listModels());
    },
  });

  async function perform(action: () => Promise<void>) {
    if (transaction.current) return;
    transaction.current = true; setPending(true); setError(null); setJob(null);
    try { await action(); } catch (err) { setError(message(err)); }
    finally { transaction.current = false; setPending(false); }
  }
  function stage(file: File, record: DocumentRecord) {
    setSettings(current => ({ ...current, prompt: "", scene_context: "" }));
    setSource(file); setDocument(record); setSize({ width: record.canvas_width, height: record.canvas_height });
    setMask(null); setMaskUrl(null); setPoints([]); setLayerPoints([]); setAlternatives([]); setProposal(null); setCompare(false);
    setOperation("replace"); setTool("object"); setExpansion(initialExpansion); setLibraryOpen(false); setJob(null);
  }
  async function openFile(file: File) {
    if (!file.type.startsWith("image/")) throw new Error("Choose a PNG, JPG or WebP image.");
    const record = await createDocument(file);
    stage(file, record); await refresh();
  }
  async function openSaved(record: DocumentRecord) {
    const current = await getDocument(record.id);
    if (!current.current_image_url) throw new Error("This document has no image.");
    stage(await fileFromUrl(fileUrl(current.current_image_url), current.name), current);
  }
  async function openGeneration(generation: GenerationRecord) {
    const record = await importGeneration(generation.id);
    stage(await fileFromUrl(fileUrl(generation.output_image_url), record.name), record);
  }
  async function travel(direction: "undo" | "redo") {
    if (!document) return;
    await layers.settled();
    const before = documentRef.current ?? document;
    const record = direction === "undo" ? await undoDocument(before.id) : await redoDocument(before.id);
    // Layer-only steps keep the same background, so the selection and prompt can stay.
    if (source && record.current_image_url === before.current_image_url) {
      documentRef.current = { ...record, name: before.name };
      setDocument(documentRef.current);
    } else await openSaved(record);
    await refresh();
  }
  function newImage() {
    setSettings(current => ({ ...current, prompt: "", scene_context: "" }));
    setSource(null); setDocument(null); setSize(null); setMask(null); setPoints([]); setMaskUrl(null);
    setAlternatives([]); setProposal(null); setOperation("generate"); setJob(null); setError(null);
    if (settings.model_name === "flux1-fill-dev") setSettings(current => ({ ...current, preset: "quality", model_name: defaultSettings.model_name, steps: defaultSettings.steps, guidance: defaultSettings.guidance }));
  }
  function chooseTool(next: CanvasTool) {
    setTool(next);
    if (operation === "expand") { clearSelection(); setOperation("replace"); }
  }
  function selectLayer(id: string | null) {
    layers.select(id);
    if (id) { setTool("transform"); if (operation === "expand") setOperation("replace"); }
  }
  async function addLayerImages(files: File[]) {
    setSideTab("layers"); collapsePanel(false);
    await layers.settled();
    await layers.addImages(files);
    setTool("transform");
    if (operation === "expand") setOperation("replace");
  }
  async function copySelectionToLayer() {
    if (!source || !mask || !size || locked) return;
    await perform(async () => {
      await layers.settled();
      const background = await createImageBitmap(source);
      const cut = await cutSelection(background, size, mask).finally(() => background.close());
      if (!cut) throw new Error("Select the area you want on its own layer first.");
      const layer = await layers.addImage(cut.blob, "Selection.png", { x: cut.box.x + cut.box.width / 2, y: cut.box.y + cut.box.height / 2 });
      if (!layer) return;
      clearSelection(); setTool("transform"); setSideTab("layers"); collapsePanel(false);
    });
  }
  function resetLayer(id: string) {
    const layer = layers.layers.find(item => item.id === id);
    if (layer && size) layers.update(id, initialPlacement(layer, size), "Reset layer");
  }
  async function invertLayerMask(id: string) {
    const layer = layers.layers.find(item => item.id === id);
    if (!layer) return;
    try {
      const current = layer.mask_url ? await loadMask(fileUrl(layer.mask_url)) : solidMask(layer.width, layer.height);
      await layers.setMask(id, invertMask(current), "Invert layer mask");
    } catch (err) { setError(message(err)); }
  }
  async function cutOutLayer(layerId: string, maskUrl: string) {
    const response = await fetch(maskUrl, { cache: "no-store" });
    if (!response.ok) throw new Error("Could not load the selection. Click the object again.");
    await layers.setMask(layerId, await response.blob(), "Cut out object");
  }
  async function layerFile(layer: LayerRecord): Promise<File> {
    let file = layerFiles.current.get(layer.image_path);
    if (!file) {
      file = await fileFromUrl(fileUrl(layer.image_url), layer.name + ".png");
      layerFiles.current.set(layer.image_path, file);
    }
    return file;
  }
  async function layerObjectPoint(point: SelectionPoint) {
    const layer = activeLayer;
    if (!layer || locked) return;
    if (!point.label && !layerPoints.length) { setError("Click the part of the layer you want to keep first."); return; }
    if (layerPoints.length >= 24) { setError("Start again to keep refining: press Esc, then click the object."); return; }
    const clamped = { ...point, x: Math.min(point.x, layer.width - 1), y: Math.min(point.y, layer.height - 1) };
    await perform(async () => {
      const next = [...layerPoints, clamped];
      await layers.settled();
      queued(await selectObject(await layerFile(layer), next), null, { layerId: layer.id }); setLayerPoints(next);
    });
  }
  async function removeLayerPoint(index: number) {
    const layer = activeLayer;
    if (!layer || locked || index < 0 || index >= layerPoints.length) return;
    const next = layerPoints.filter((_, position) => position !== index);
    // Without a keep point there is nothing to cut out; leave the layer as it is.
    if (!next.some(point => point.label)) { setLayerPoints([]); return; }
    await perform(async () => { queued(await selectObject(await layerFile(layer), next), null, { layerId: layer.id }); setLayerPoints(next); });
  }
  function chooseExpansion() {
    if (!models.find(model => model.name === settings.model_name)?.capabilities.includes("outpaint")) {
      const config = GENERATION_PRESETS.fast;
      setSettings(current => ({ ...current, preset: "fast", model_name: config.modelName, steps: config.steps, guidance: config.guidance }));
    }
    setOperation("expand");
  }
  function chooseReference(id: number) {
    const wasActive = refs.active.some(item => item.id === id);
    if (!refs.toggle(id)) { setError(`Use up to ${MAX_ACTIVE_REFERENCES} references per edit. Take one out first.`); return; }
    if (wasActive) return;
    if (operation !== "replace") setOperation("replace");
    // References need FLUX.2 Klein; two proposals make a good likeness easy to pick.
    setSettings(current => {
      const quality = isKleinModel(current.model_name) ? {} : { preset: "quality" as const, model_name: GENERATION_PRESETS.quality.modelName, steps: GENERATION_PRESETS.quality.steps, guidance: GENERATION_PRESETS.quality.guidance };
      return { ...current, ...quality, count: current.count === 1 ? 2 : current.count };
    });
  }
  function collapsePanel(collapsed: boolean) {
    setPanelCollapsed(collapsed);
    try { localStorage.setItem(PANEL_COLLAPSED_KEY, collapsed ? "1" : "0"); } catch { /* Preference only. */ }
  }
  function clearSelection() { setClearToken(value => value + 1); setMask(null); setMaskUrl(null); setPoints([]); }
  function queued(created: JobResponse, operationType: JobContext["operation"], extras: JobExtras = {}) {
    cancellation.current = null;
    const current = documentRef.current;
    context.current = {
      jobId: created.id, documentId: current?.id ?? null, revisionId: current?.current_revision_id ?? null, operation: operationType,
      layerId: extras.layerId ?? null, layerOffset: extras.layerOffset ?? { x: 0, y: 0 },
    };
    setJob({ ...created, progress: 0, error: null, generation_id: null, generation_ids: [] });
  }
  async function objectPoint(point: SelectionPoint) {
    if (!source || locked) return;
    if (activeLayer) { await layerObjectPoint(point); return; }
    if (!point.label && !points.length) { setError("Click the object you want to select first."); return; }
    if (points.length >= 24) { setError("Start a new selection to continue refining."); return; }
    await perform(async () => {
      const next = [...points, point];
      queued(await selectObject(source, next), null); setPoints(next);
    });
  }
  async function removePoint(index: number) {
    if (!source || locked || index < 0 || index >= points.length) return;
    const next = points.filter((_, position) => position !== index);
    // Without an include point, the remaining exclusions select nothing.
    if (!next.some(point => point.label)) { clearSelection(); return; }
    await perform(async () => { queued(await selectObject(source, next), null); setPoints(next); });
  }
  async function submit() {
    if (locked || !online || cudaBlocked) return;
    await perform(async () => {
      await layers.settled();
      let prompt = settings.prompt.trim();
      if (operation === "remove") {
        prompt = isKleinModel(settings.model_name)
          ? prompt || "Continue the surrounding background."
          : "Remove the selected object completely. Reconstruct the background naturally, matching the surrounding lighting, texture and perspective. " + prompt;
      }
      const parameters = { prompt, negative_prompt: settings.negative_prompt || undefined, model_name: settings.model_name, runtime_mode: settings.runtime_mode, seed: settings.seed ? Number(settings.seed) : undefined, steps: settings.steps, guidance: settings.guidance, count: settings.count };
      if (task === "text-to-image") queued(await generateImage({ ...parameters, width: settings.width, height: settings.height }), null);
      else if (task === "outpaint" && source && size && validExpansion(size, expansion)) queued(await outpaintImage(source, { ...parameters, ...expansion }), "outpaint", { layerOffset: { x: expansion.left, y: expansion.top } });
      else if (source && mask && document) {
        const form = new FormData();
        for (const [key, value] of Object.entries(parameters)) if (value !== undefined) form.set(key, String(value));
        form.set("strength", String(settings.strength)); form.set("image", source); form.set("mask", mask, "mask.png");
        form.set("edit_operation", operation === "remove" ? "remove" : "replace");
        // The canvas sends the final soft mask (Expand/Soften), so no extra server-side ramp.
        form.set("blend_feather", "0");
        form.set("context_padding", String(settings.context_padding));
        form.set("scene_context", settings.scene_context);
        for (const item of references) form.append("reference_ids", String(item.id));
        queued(await inpaintImage(form), "inpaint");
      } else throw new Error("Select the area you want to change first.");
    });
  }
  async function apply() {
    await perform(async () => {
      if (document && proposal) {
        const result = await acceptAlternative(document.id, proposal.id, proposal.alternatives[selected].id);
        await openSaved(result); await refresh();
      } else if (chosen) { await openGeneration(chosen); await refresh(); }
    });
  }
  function discard() { setAlternatives([]); setProposal(null); setCompare(false); }
  async function cancel() {
    if (!job || !jobActive || cancelling || cancellation.current) return;
    const { id, status } = job;
    cancellation.current = { id, requestedAt: Date.now() };
    setError(null);
    setJob(current => current?.id === id ? { ...current, status: "cancelling" } : current);
    try {
      const next = await cancelJob(id);
      if (cancellation.current?.id === id) setJob(next);
    } catch (err) {
      if (cancellation.current?.id !== id) return;
      cancellation.current = null;
      setJob(current => current?.id === id ? { ...current, status } : current);
      setError("Could not cancel the operation. " + message(err));
    }
  }
  function onFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file && !locked) void perform(() => openFile(file));
    event.target.value = "";
  }
  function drop(event: DragEvent<HTMLElement>) {
    event.preventDefault(); setDropActive(false);
    const file = Array.from(event.dataTransfer.files).find(item => item.type.startsWith("image/"));
    if (file && !locked) void perform(() => openFile(file));
  }
  async function exportImage() {
    if (!sourceUrl || !source) return;
    const visible = layers.layers.filter(layer => layer.visible && layer.opacity > 0);
    let href = sourceUrl;
    let extension = source.type === "image/jpeg" ? ".jpg" : source.type === "image/webp" ? ".webp" : ".png";
    // Layers are flattened onto the picture, as PNG to keep every edge crisp.
    if (visible.length && size) {
      const background = await createImageBitmap(source);
      try {
        const parts = await Promise.all(visible.map(async layer => ({
          ...layer, image: await loadImage(fileUrl(layer.image_url)), mask: layer.mask_url ? await loadMask(fileUrl(layer.mask_url)) : null,
        })));
        href = URL.createObjectURL(await flatten(background, size, parts));
        extension = ".png";
      } finally { background.close(); }
    }
    const anchor = window.document.createElement("a"); anchor.href = href;
    anchor.download = (document?.name ?? "MosImage").replace(/[<>:"/\\|?*]+/g, "-").replace(/\.[^.]+$/, "") + extension;
    anchor.click();
    if (href !== sourceUrl) window.setTimeout(() => URL.revokeObjectURL(href), 1000);
  }

  const expansionValid = !!size && validExpansion(size, expansion);
  const canSubmit = !locked && online && !cudaBlocked && (task === "outpaint"
    ? expansionValid && !!settings.prompt.trim()
    : (operation === "remove" || !!settings.prompt.trim()) && (task === "text-to-image" || !!mask) && !referenceNeedsQuality);
  const hint = activeLayer ? `Editing layer “${activeLayer.name}”. AI edits change the background; press Esc to go back to it.`
    : referenceNeedsQuality ? "References need the Quality model. Choose Quality in generation settings."
    : operationHint(operation, !!source, !!mask, tool, references.map(item => item.name));
  const referenceNote = operation === "expand" ? "References are used for Replace edits. Leave Expand to place one."
    : operation === "remove" && refs.active.length ? "Switch to Replace to place the selected references." : null;

  return <main className={"focus-app" + (document ? " with-references" : "")}>
    <TopBar document={document} onRename={name => document && setDocument({ ...document, name })} online={online} connecting={connecting} system={system} warmUp={warmUp} locked={locked} canExport={!!source}
      onUndo={() => void perform(() => travel("undo"))} onRedo={() => void perform(() => travel("redo"))}
      onOpen={() => upload.current?.click()} onLibrary={() => setLibraryOpen(true)} onExport={() => void exportImage().catch(err => setError(message(err)))} />
    <input ref={upload} className="visually-hidden" type="file" accept="image/png,image/jpeg,image/webp" onChange={onFile} />
    <ToolRail hasImage={!!source} tool={tool} expanding={operation === "expand"} locked={locked} onNew={newImage} onTool={chooseTool} onExpand={chooseExpansion} />
    <section className="focus-workspace" onDragOver={event => { event.preventDefault(); if (!locked) setDropActive(true); }} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDropActive(false); }} onDrop={drop}>
      <div className="canvas-area">
        {source && sourceUrl ? operation === "expand" && size
          ? <ExpansionCanvas imageUrl={sourceUrl} size={size} expansion={expansion} onChange={setExpansion} disabled={locked} reviewing={reviewing} previewUrl={previewUrl} previewSize={chosen?.width && chosen.height ? { width: chosen.width, height: chosen.height } : null} />
          : <EditorCanvas imageUrl={sourceUrl} previewImageUrl={previewUrl} reviewing={reviewing} imageName={document?.name} onMaskChange={setMask} tool={tool} disabled={locked} maskImportUrl={maskUrl} clearToken={clearToken}
            points={points} onObjectPoint={point => void objectPoint(point)} onRemovePoint={index => void removePoint(index)} onClearSelection={clearSelection}
            onImageSize={setSize} onError={setError} contextPadding={settings.context_padding} onContextPadding={context_padding => setSettings(current => ({ ...current, context_padding }))} onHelp={() => setPanel("shortcuts")}
            layerActive={!!activeLayer} layerPointCount={layerPoints.length} onUndoLayerPoint={() => void removeLayerPoint(layerPoints.length - 1)}
            renderLayers={view => <LayerStage layers={layers.layers} active={activeLayer} zoom={view.zoom} size={view.size} tool={tool} brushSize={view.brushSize} hide={view.hide}
              interactive={view.interactive} disabled={locked} points={layerPoints} onSelect={layers.select}
              onPreview={layers.preview} onCancelPreview={layers.cancelPreview} onCommit={layers.update} onMask={(id, layerMask, label) => void layers.setMask(id, layerMask, label)}
              onObjectPoint={point => void layerObjectPoint(point)} onRemovePoint={index => void removeLayerPoint(index)} onCursor={view.trackCursor} />} />
          : previewUrl ? <div className="generated-preview"><img crossOrigin="anonymous" src={previewUrl} alt={chosen.prompt} /><span>{chosen.width} × {chosen.height}</span></div>
          : <WelcomeCanvas disabled={locked} onOpen={() => upload.current?.click()} />}
        {busy && <JobOverlay job={job} />}
        {dropActive && <div className="drop-overlay">Drop to open your image</div>}
      </div>
      {reviewing
        ? <ProposalTray thumbnails={proposalUrls} selected={selected} onSelect={index => { setSelected(index); setCompare(false); }} canCompare={!!source} comparing={compare} onCompare={() => setCompare(value => !value)} busy={busy} onDiscard={discard} onApply={() => void apply()} />
        : <CommandCard operation={operation} onOperation={setOperation} hasImage={!!source} locked={locked}
          settingsLabel={settingsSummary(settings, task)} prompt={settings.prompt} onPrompt={prompt => setSettings({ ...settings, prompt })}
          references={references.map(item => ({ id: item.id, name: item.name, imageUrl: fileUrl(item.subject_url ?? item.image_url) }))} onRemoveReference={id => refs.toggle(id)}
          hint={hint} cancelled={job?.status === "cancelled"} jobActive={jobActive} cancelling={cancelling} canSubmit={canSubmit}
          onCanvasSize={() => setPanel("expansion")} onSettings={() => setPanel("settings")} onSubmit={() => void submit()} onCancel={() => void cancel()} />}
      {(cudaBlocked || error) && <div className="error-notice" role="alert">
        <span>{error ?? system?.setup_hint ?? "CUDA is unavailable. Check the local PyTorch installation."}</span>
        {!online && !connecting && <button onClick={() => void connect()}>Reconnect</button>}
        {error && <button className="icon-button" aria-label="Dismiss message" onClick={() => setError(null)}><Icon name="close" /></button>}
      </div>}
    </section>
    {document && <SidePanel tab={sideTab} onTab={setSideTab} collapsed={panelCollapsed} onCollapsed={collapsePanel} layerCount={layers.layers.length} referenceCount={refs.references.length}>
      {sideTab === "layers"
        ? <LayersPanel layers={layers.layers} activeId={activeLayer?.id ?? null} backgroundUrl={sourceUrl} uploading={layers.uploading} disabled={locked}
          canCopySelection={!!mask} onSelect={selectLayer} onAdd={files => void addLayerImages(files)} onCopySelection={() => void copySelectionToLayer()}
          onUpdate={layers.update} onPreview={layers.preview} onRemove={layers.remove} onDuplicate={layers.duplicate} onReorder={layers.reorder}
          onReset={resetLayer} onInvertMask={id => void invertLayerMask(id)} />
        : <ReferencePanel references={refs.references} activeIds={refs.active.map(item => item.id)} pending={refs.pending} uploading={refs.uploading}
          disabled={locked || operation === "expand"} note={referenceNote}
          onUse={chooseReference} onAdd={files => void refs.add(files).then(id => { if (id !== null) chooseReference(id); })} onRemove={id => void refs.remove(id)} onIsolate={(id, isolated) => void refs.setIsolated(id, isolated)} />}
    </SidePanel>}
    {panel === "settings" && <ControlDialog title="Generation settings" description="Choose how your image is created." onClose={() => setPanel(null)}><SettingsControls settings={settings} onChange={setSettings} models={models} task={task} disabled={locked} /></ControlDialog>}
    {panel === "expansion" && size && <ExpansionDialog size={size} expansion={expansion} onChange={setExpansion} disabled={locked} onClose={() => setPanel(null)} />}
    {panel === "shortcuts" && <ShortcutsDialog onClose={() => setPanel(null)} />}
    {libraryOpen && <LibraryDrawer documents={documents} history={history} onClose={() => setLibraryOpen(false)} onDocument={item => void perform(() => openSaved(item))} onGeneration={item => void perform(() => openGeneration(item))} />}
  </main>;
}

function message(error: unknown): string { return error instanceof Error ? error.message : "The operation failed."; }
async function fileFromUrl(url: string, name: string): Promise<File> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error("Could not open the saved image.");
  const blob = await response.blob();
  return new File([blob], name, { type: blob.type || "image/png" });
}
