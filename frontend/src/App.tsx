import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import {
  DocumentRecord, Expansion, GenerationRecord, JobStatus, RevisionRecord,
  acceptAlternative, copyReferences, createDocument, deleteDocument, deleteGeneration, fileUrl, getDocument, getGeneration,
  importGeneration, listModels, proposeAlternatives, redoDocument, undoDocument, updateDocument,
} from "./api";
import { EditorCanvas } from "./canvas/EditorCanvas";
import { CanvasTool, isSelectionTool, toolForKey } from "./canvas/tools";
import { ImageSize } from "./canvas/coordinates";
import { ExpansionCanvas } from "./canvas/ExpansionCanvas";
import { validExpansion } from "./canvas/expansion";
import { errorMessage } from "./errors";
import { EditScope, Operation, SketchScope, isWholeImage, taskFor } from "./generation/operation";
import { CommandState, canSubmit, commandHint } from "./generation/commandState";
import { LayerResult, submitJob } from "./generation/submission";
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
import { canvasBlob, cutSelection, fileFromUrl, invertMask, layerFootprint, loadImage, loadMask, selectionPatch, solidMask } from "./layers/bitmaps";
import { generationPlacement, initialPlacement } from "./layers/geometry";
import { useLayers } from "./hooks/useLayers";
import { MAX_ACTIVE_REFERENCES, useReferences } from "./hooks/useReferences";
import { useEngine } from "./hooks/useEngine";
import { JobContext, useJobRunner } from "./hooks/useJobRunner";
import { editedLayer, usePicture } from "./hooks/usePicture";
import { useResultPreview } from "./hooks/useResultPreview";
import { useSelectionTools } from "./hooks/useSelectionTools";
import { isTyping, useWindowEvents } from "./hooks/useWindowEvents";

const initialExpansion: Expansion = { left: 128, top: 0, right: 128, bottom: 0 };
const PANEL_COLLAPSED_KEY = "mosimage.sidePanelCollapsed";
const BLEND_PROMPT = "Blend it naturally into the scene: match the lighting, colours, shadows, reflections, perspective and grain, so it looks photographed here. Keep its shape and details.";

export default function App() {
  const [settings, setSettings] = useState(defaultSettings);
  const [error, setError] = useState<string | null>(null);
  const engine = useEngine(setError);
  const { models, setModels, system, online, cudaBlocked, refresh } = engine;
  const [document, setDocument] = useState<DocumentRecord | null>(null);
  const [pending, setPending] = useState(false);
  const [source, setSource] = useState<File | null>(null);
  const [size, setSize] = useState<ImageSize | null>(null);
  const [tool, setTool] = useState<CanvasTool>("object");
  // Mask layer points the selection tools at the selected layer's mask instead of the picture.
  const [layerMasking, setLayerMasking] = useState(false);
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
  const [blendTarget, setBlendTarget] = useState<string | null>(null);
  const [layerResult, setLayerResult] = useState<LayerResult | null>(null);
  const [editScope, setEditScope] = useState<EditScope>("visible");
  const [sketch, setSketch] = useState<Blob | null>(null);
  const [sketchClear, setSketchClear] = useState(0);
  const [sketchScope, setSketchScope] = useState<SketchScope>("strokes");
  /** Start-screen references behind the proposals under review. */
  const generationReferences = useRef<number[]>([]);
  const refs = useReferences(document?.id ?? null, setError);
  // Layer saves move the revision on before React re-renders; requests read it here.
  const documentRef = useRef<DocumentRecord | null>(document);
  documentRef.current = document;
  const layers = useLayers(document, setDocument, documentRef, setError);
  const upload = useRef<HTMLInputElement>(null);
  const transaction = useRef(false);
  const picture = usePicture(source, size, sourceUrl, documentRef);
  const jobs = useJobRunner({ documentRef, setError, onDisconnected: () => engine.setOnline(false), onCompleted: jobCompleted });
  const { job } = jobs;

  const busy = pending || jobs.active;
  const reviewing = alternatives.length > 0;
  const locked = busy || reviewing;
  const chosen = alternatives[selected];
  const proposalUrls = alternatives.map((item, index) => fileUrl(proposal?.alternatives[index]?.output_image_url ?? item.output_image_url));
  const task = taskFor(operation);
  // References place people in a selection (Replace), the whole picture (Edit) or a new image (Generate).
  const references = ((operation === "replace" || operation === "edit" || operation === "faces") && source) || operation === "generate" ? refs.active : [];
  // Selection tools select on the picture, as in Photoshop, unless the layer is being masked.
  const activeLayer = operation === "expand" || operation === "sketch" || isWholeImage(operation) || (isSelectionTool(tool) && !layerMasking) ? null : layers.active;
  // The layer Edit works on alone; it stays selected without its transform handles.
  const editLayer = operation === "edit" && editScope === "layer" ? layers.active : null;
  const blending = layers.layers.find(layer => layer.id === blendTarget && layer.visible) ?? null;
  const selection = useSelectionTools({
    source, locked, activeLayer, layers, perform, queued: jobs.queued, picture: picture.editImage, layerFile: picture.layerFile, setError,
    onClear: () => setBlendTarget(null),
  });
  const { mask } = selection;
  const generationPreviewUrl = useResultPreview({ result: layerResult, chosen, source, size, documentRef, picture: picture.editImage, onError: setError });
  // Full generations and layer edits preview as Apply places them; regional results already show the whole picture.
  const previewUrl = chosen && !compare ? layerResult && (layerResult.kind === "generation" || layerResult.kind === "layer") && document
    ? generationPreviewUrl : proposalUrls[selected] : null;

  useEffect(() => { setLayerMasking(false); }, [layers.activeId]);

  useEffect(() => {
    const url = source ? URL.createObjectURL(source) : null;
    setSourceUrl(url);
    return () => { if (url) URL.revokeObjectURL(url); };
  }, [source]);

  // Debounced autosave of the document name.
  useEffect(() => {
    if (!document || !document.name.trim()) return;
    const { id, name } = document;
    const timer = window.setTimeout(() => {
      void updateDocument(id, name).then(saved => {
        setDocument(current => current?.id === id && current.name === name ? { ...current, updated_at: saved.updated_at } : current);
        engine.setDocuments(current => current.map(item => item.id === id ? { ...item, name: saved.name } : item));
      }).catch(err => setError(errorMessage(err)));
    }, 650);
    return () => window.clearTimeout(timer);
  }, [document?.id, document?.name]);

  useWindowEvents({ onKey: shortcut, onPaste: paste });

  function paste(event: ClipboardEvent) {
    if (locked) return;
    const file = Array.from(event.clipboardData?.files ?? []).find(item => item.type.startsWith("image/"));
    if (!file) return;
    event.preventDefault();
    // With a picture open, a pasted image goes on top of it as a layer.
    if (document && source) void addLayerImages([file]);
    else void perform(() => openFile(file));
  }
  function shortcut(event: KeyboardEvent) {
    if (isTyping(event)) return;
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
      const layerPoints = selection.layerPoints;
      if (arrows[event.key]) { event.preventDefault(); layers.nudge(activeLayer.id, arrows[event.key][0] * step, arrows[event.key][1] * step); return; }
      if (event.key === "Backspace" && tool === "object" && layerPoints.length) { event.preventDefault(); void selection.removeLayerPoint(layerPoints.length - 1); return; }
      if (event.key === "Delete" || event.key === "Backspace") { event.preventDefault(); layers.remove(activeLayer.id); return; }
      if (event.key === "Escape") { if (layerPoints.length) selection.setLayerPoints([]); else if (layerMasking) finishMasking(); else layers.select(null); return; }
    }
    if (ctrl && key === "j" && mask && source) { event.preventDefault(); void copySelectionToLayer(); return; }
    if (ctrl && key === "o") { event.preventDefault(); upload.current?.click(); }
    if (ctrl && key === "z" && document && tool !== "sketch") {
      event.preventDefault();
      if (event.shiftKey ? document.can_redo : document.can_undo) void perform(() => travel(event.shiftKey ? "redo" : "undo"));
    }
    if (!ctrl && !event.altKey && source) {
      if (key === "c") { chooseExpansion(); return; }
      const next = toolForKey(key);
      if (next) chooseTool(next);
    }
  }

  async function jobCompleted(next: JobStatus, submitted: JobContext | null, isDisposed: () => boolean) {
    if (next.mask_image_url) {
      if (submitted?.layerId) await selection.cutOutLayer(submitted.layerId, fileUrl(next.mask_image_url), submitted.maskLabel);
      else selection.setMaskUrl(fileUrl(next.mask_image_url));
    }
    if (next.generation_ids.length) {
      const generated = await Promise.all(next.generation_ids.map(getGeneration));
      const revision = submitted?.documentId && submitted.operation
        ? await proposeAlternatives(submitted.documentId, next.generation_ids, submitted.operation, submitted.revisionId ?? undefined, submitted.layerOffset) : null;
      if (isDisposed()) return;
      setAlternatives(generated); setProposal(revision); setLayerResult(submitted?.result ?? null); setSelected(0); setCompare(false);
      await refresh();
    }
    setModels(await listModels());
  }

  async function perform(action: () => Promise<void>) {
    if (transaction.current) return;
    transaction.current = true; setPending(true); setError(null); jobs.setJob(null);
    try { await action(); } catch (err) { setError(errorMessage(err)); }
    finally { transaction.current = false; setPending(false); }
  }
  function stage(file: File, record: DocumentRecord) {
    setSettings(current => ({ ...current, prompt: "", scene_context: "" }));
    documentRef.current = record;
    setSource(file); setDocument(record); setSize({ width: record.canvas_width, height: record.canvas_height });
    selection.reset(); setAlternatives([]); setProposal(null); setCompare(false);
    setBlendTarget(null); setLayerResult(null);
    setOperation("replace"); setTool("object"); setExpansion(initialExpansion); setLibraryOpen(false); jobs.setJob(null);
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
  /** Opens a result as a document; start-screen references it used are copied along. */
  async function openGeneration(generation: GenerationRecord, referenceIds: number[] = []) {
    const record = await importGeneration(generation.id);
    if (!record.current_image_url) throw new Error("This document has no image.");
    if (referenceIds.length) await copyReferences(record.id, referenceIds);
    stage(await fileFromUrl(fileUrl(record.current_image_url), record.name), record);
    setSideTab("layers"); collapsePanel(false);
  }
  /** A white canvas at the chosen size, ready to sketch on. */
  async function openBlank() {
    const canvas = window.document.createElement("canvas");
    canvas.width = settings.width; canvas.height = settings.height;
    const context = canvas.getContext("2d")!;
    context.fillStyle = "white"; context.fillRect(0, 0, canvas.width, canvas.height);
    await openFile(new File([await canvasBlob(canvas)], "Sketch.png", { type: "image/png" }));
    setSketchScope("whole"); chooseOperation("sketch");
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
    documentRef.current = null;
    setSource(null); setDocument(null); setSize(null); selection.reset();
    setAlternatives([]); setProposal(null); setLayerResult(null); setBlendTarget(null); setOperation("generate"); jobs.setJob(null); setError(null);
  }

  function chooseEditScope(next: EditScope) {
    setEditScope(next);
    if (next === "visible") { layers.select(null); return; }
    // Start on the top layer; the Layers panel picks another.
    if (!layers.active && layers.layers.length) layers.select(layers.layers[layers.layers.length - 1].id);
    setSideTab("layers"); collapsePanel(false);
  }
  function chooseTool(next: CanvasTool) {
    if (next === "sketch") { chooseOperation("sketch"); return; }
    setTool(next);
    if (operation === "sketch" && next !== "move") setOperation("replace");
    if (operation === "expand" || isWholeImage(operation)) { selection.clear(); setOperation("replace"); }
    if (next === "transform") setLayerMasking(false);
    else if (isSelectionTool(next) && !layerMasking) layers.select(null);
  }
  function toggleLayerMask(id: string) {
    if (layerMasking) { finishMasking(); return; }
    layers.select(id); setLayerMasking(true);
    setTool(current => isSelectionTool(current) ? current : "object");
    if (operation === "expand" || operation === "sketch" || isWholeImage(operation)) { selection.clear(); setOperation("replace"); }
  }
  function finishMasking() { setLayerMasking(false); selection.setLayerPoints([]); setTool("transform"); }
  function chooseOperation(next: Exclude<Operation, "expand">) {
    // The pencil belongs to Sketch; other operations go back to selecting.
    if (next === "sketch") { layers.select(null); setLayerMasking(false); setTool("sketch"); }
    else if (tool === "sketch" || (next === "faces" && tool === "transform")) setTool("object");
    // Faces work on the picture as you see it; a selection may narrow them down.
    if (next === "faces") { layers.select(null); setLayerMasking(false); }
    if (isWholeImage(next)) {
      // With a layer selected, Edit starts on that layer alone.
      if (next === "edit") setEditScope(layers.active ? "layer" : "visible");
      else layers.select(null);
      selection.clear();
    }
    setOperation(next);
  }
  function chooseExpansion() { setOperation("expand"); }
  function chooseReference(id: number) {
    const wasActive = refs.active.some(item => item.id === id);
    if (!refs.toggle(id)) { setError(`Use up to ${MAX_ACTIVE_REFERENCES} references per edit. Take one out first.`); return; }
    if (wasActive) return;
    // Without a picture, references compose a new image.
    if (!source) setOperation("generate");
    else if (operation !== "replace" && !isWholeImage(operation)) setOperation("replace");
    // Two proposals make a good likeness easy to pick.
    setSettings(current => ({ ...current, count: current.count === 1 ? 2 : current.count }));
  }
  function collapsePanel(collapsed: boolean) {
    setPanelCollapsed(collapsed);
    try { localStorage.setItem(PANEL_COLLAPSED_KEY, collapsed ? "1" : "0"); } catch { /* Preference only. */ }
  }

  function selectLayer(id: string | null) {
    layers.select(id);
    if (operation === "edit") { setEditScope(id ? "layer" : "visible"); return; }
    if (id) { setTool("transform"); if (operation === "expand" || operation === "sketch" || isWholeImage(operation)) setOperation("replace"); }
  }
  async function addLayerImages(files: File[]) {
    setSideTab("layers"); collapsePanel(false);
    await layers.settled();
    await layers.addImages(files);
    setTool("transform");
    if (operation === "expand" || operation === "sketch" || isWholeImage(operation)) setOperation("replace");
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
      selection.clear(); setTool("transform"); setSideTab("layers"); collapsePanel(false);
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
    } catch (err) { setError(errorMessage(err)); }
  }
  /** Selects where a layer shows, a little wider, so the AI can redraw it to fit the scene. */
  async function blendLayerIn(id: string) {
    const layer = layers.layers.find(item => item.id === id);
    if (!layer || !size || locked) return;
    if (!layer.visible) { setError("Show the layer first, then blend it in."); return; }
    await perform(async () => {
      await layers.settled();
      const image = await loadImage(fileUrl(layer.image_url));
      const layerMask = layer.mask_url ? await loadMask(fileUrl(layer.mask_url)) : null;
      const reach = Math.max(layer.width * Math.abs(layer.scale_x), layer.height * Math.abs(layer.scale_y));
      const grow = Math.round(Math.min(48, Math.max(8, reach * 0.06)));
      const footprint = await layerFootprint(size, { ...layer, image, mask: layerMask }, grow, grow / 2);
      layers.select(null); selection.show(URL.createObjectURL(footprint));
      setBlendTarget(id); setOperation("replace"); setTool("brush");
      setSettings(current => ({ ...current, prompt: BLEND_PROMPT }));
    });
  }

  async function submit() {
    if (locked || !online || cudaBlocked) return;
    await perform(async () => {
      await layers.settled();
      const submission = await submitJob({
        operation, settings, referenceIds: references.map(item => item.id), source, size, hasDocument: !!document, layerCount: layers.layers.length,
        mask, sketch, sketchScope, editScope, editLayer, expansion, picture: picture.editImage,
        blendLayer: picture.visibleLayers().find(layer => layer.id === blendTarget) ?? null,
      });
      if (submission.startReferences) generationReferences.current = submission.startReferences;
      jobs.queued(submission.created, submission.operation, submission.extras);
    });
  }
  async function apply() {
    await perform(async () => {
      if (layerResult && chosen) await applyAsLayer(layerResult, proposalUrls[selected]);
      else if (document && proposal) {
        const result = await acceptAlternative(document.id, proposal.id, proposal.alternatives[selected].id);
        await openSaved(result); await refresh();
      } else if (chosen) {
        // Only references still on the start screen can come along.
        const used = generationReferences.current.filter(id => refs.references.some(item => item.id === id));
        await openGeneration(chosen, used); await refresh();
      }
    });
  }
  function discard() { setAlternatives([]); setProposal(null); setLayerResult(null); setCompare(false); }
  async function applyAsLayer(result: LayerResult, url: string) {
    if (!size) return;
    await layers.settled();
    if (result.kind === "generation") {
      const file = await fileFromUrl(url, "generation.png");
      const image = await loadImage(url);
      const placement = generationPlacement({ width: image.naturalWidth, height: image.naturalHeight }, size);
      if (!await layers.addResult(file, null, result.name, placement, null)) return;
    } else if (result.kind === "layer") {
      const layer = layers.layers.find(item => item.id === result.layerId);
      if (!layer) throw new Error("That layer was deleted, so the edit has nowhere to go.");
      const image = await canvasBlob(await editedLayer(await loadImage(url), layer));
      const layerMask = layer.mask_url ? await fileFromUrl(fileUrl(layer.mask_url), "mask.png") : null;
      if (!await layers.addResult(image, layerMask, result.name, layer, layer.id, "Edit layer")) return;
    } else {
      // Fixed faces keep their own mask: only the faces become the layer.
      const footprint = result.kind === "faces" ? await facesMask() : result.mask;
      const patch = await selectionPatch(await loadImage(url), size, footprint);
      if (!patch) throw new Error("The selection was empty, so there is nothing to add.");
      const center = { x: patch.box.x + patch.box.width / 2, y: patch.box.y + patch.box.height / 2 };
      if (!await layers.addResult(patch.image, patch.mask, result.name, center, result.kind === "selection" ? result.replaces : null)) return;
    }
    await layers.settled();
    discard(); selection.clear(); setSketchClear(value => value + 1); setOperation("replace"); setTool("transform"); setSideTab("layers"); collapsePanel(false);
    await refresh();
  }
  async function facesMask(): Promise<Blob> {
    if (!chosen?.mask_image_path) throw new Error("The faces mask is missing. Fix the faces again.");
    const response = await fetch(fileUrl("/files/" + chosen.mask_image_path), { cache: "no-store" });
    if (!response.ok) throw new Error("Could not load the faces mask. Fix the faces again.");
    return response.blob();
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

  const command: CommandState = {
    operation, prompt: settings.prompt, hasImage: !!source, hasMask: !!mask, hasSketch: !!sketch,
    expansionValid: !!size && validExpansion(size, expansion), editScope, editLayerName: editLayer?.name ?? null, sketchScope,
    referenceNames: references.map(item => item.name), tool, activeLayerName: activeLayer?.name ?? null, layerMasking, blendingName: blending?.name ?? null,
  };
  const submittable = !locked && online && !cudaBlocked && canSubmit(command);
  const referenceNote = operation === "expand" ? "References are used for Replace edits. Leave Expand to place one."
    : operation === "remove" && refs.active.length ? "Switch to Replace to place the selected references." : null;

  return <main className="focus-app with-references">
    <TopBar document={document} onRename={name => document && setDocument({ ...document, name })} online={online} connecting={engine.connecting} system={system} warmUp={engine.warmUp} locked={locked} canExport={!!source}
      onUndo={() => void perform(() => travel("undo"))} onRedo={() => void perform(() => travel("redo"))}
      onOpen={() => upload.current?.click()} onLibrary={() => setLibraryOpen(true)} onExport={() => void picture.exportPicture(document?.name ?? "MosImage").catch(err => setError(errorMessage(err)))} />
    <input ref={upload} className="visually-hidden" type="file" accept="image/png,image/jpeg,image/webp" onChange={onFile} />
    <ToolRail hasImage={!!source} tool={tool} expanding={operation === "expand"} locked={locked} onNew={newImage} onTool={chooseTool} onExpand={chooseExpansion} />
    <section className="focus-workspace" onDragOver={event => { event.preventDefault(); if (!locked) setDropActive(true); }} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDropActive(false); }} onDrop={drop}>
      <div className="canvas-area">
        {source && sourceUrl ? operation === "expand" && size
          ? <ExpansionCanvas imageUrl={sourceUrl} size={size} expansion={expansion} onChange={setExpansion} disabled={locked} reviewing={reviewing} previewUrl={previewUrl} previewSize={chosen?.width && chosen.height ? { width: chosen.width, height: chosen.height } : null} />
          : <EditorCanvas imageUrl={sourceUrl} previewImageUrl={previewUrl} reviewing={reviewing} imageName={document?.name} onMaskChange={selection.setMask} tool={tool} disabled={locked} maskImportUrl={selection.maskUrl} clearToken={selection.clearToken}
            points={selection.points} onObjectPoint={point => void selection.objectPoint(point)} onRemovePoint={index => void selection.removePoint(index)} onClearSelection={selection.clear}
            onImageSize={setSize} onError={setError} contextPadding={settings.context_padding} onContextPadding={context_padding => setSettings(current => ({ ...current, context_padding }))} onHelp={() => setPanel("shortcuts")}
            layerActive={!!activeLayer} layerPointCount={selection.layerPoints.length} onUndoLayerPoint={() => void selection.removeLayerPoint(selection.layerPoints.length - 1)}
            sketchVisible={operation === "sketch"} sketchClearToken={sketchClear} onSketchChange={setSketch}
            onFind={text => void selection.findByText(text)} onSubject={() => void selection.selectSubject()}
            renderLayers={view => previewUrl ? null : <LayerStage layers={layers.layers} active={activeLayer} zoom={view.zoom} size={view.size} tool={tool} brushSize={view.brushSize} hide={view.hide}
              interactive={view.interactive} disabled={locked} points={selection.layerPoints} onSelect={layers.select}
              onPreview={layers.preview} onCancelPreview={layers.cancelPreview} onCommit={layers.update} onMask={(id, layerMask, label) => void layers.setMask(id, layerMask, label)}
              onObjectPoint={point => void selection.objectPoint(point)} onRemovePoint={index => void selection.removeLayerPoint(index)} onCursor={view.trackCursor} />} />
          : previewUrl ? <div className="generated-preview"><img crossOrigin="anonymous" src={previewUrl} alt={chosen.prompt} /><span>{chosen.width} × {chosen.height}</span></div>
          : <WelcomeCanvas disabled={locked || !online} onOpen={() => upload.current?.click()} onBlank={() => void perform(openBlank)} />}
        {busy && <JobOverlay job={job} />}
        {dropActive && <div className="drop-overlay">Drop to open your image</div>}
      </div>
      {reviewing
        ? <ProposalTray thumbnails={proposalUrls} selected={selected} onSelect={index => { setSelected(index); setCompare(false); }} canCompare={!!source} comparing={compare} onCompare={() => setCompare(value => !value)} busy={busy} onDiscard={discard} onApply={() => void apply()} />
        : <CommandCard operation={operation} onOperation={chooseOperation} hasImage={!!source} locked={locked}
          editScope={editScope} onEditScope={document && layers.layers.length ? chooseEditScope : undefined} sketchScope={sketchScope} onSketchScope={setSketchScope}
          settingsLabel={settingsSummary(settings, task)} prompt={settings.prompt} onPrompt={prompt => setSettings({ ...settings, prompt })}
          references={references.map(item => ({ id: item.id, name: item.name, imageUrl: fileUrl(item.subject_url ?? item.image_url) }))} onRemoveReference={id => refs.toggle(id)}
          hint={commandHint(command)} cancelled={job?.status === "cancelled"} jobActive={jobs.active} cancelling={jobs.cancelling} canSubmit={submittable}
          onCanvasSize={() => setPanel("expansion")} onSettings={() => setPanel("settings")} onSubmit={() => void submit()} onCancel={() => void jobs.cancel()} />}
      {(cudaBlocked || error) && <div className="error-notice" role="alert">
        <span>{error ?? system?.setup_hint ?? "CUDA is unavailable. Check the local PyTorch installation."}</span>
        {!online && !engine.connecting && <button onClick={() => void engine.connect()}>Reconnect</button>}
        {error && <button className="icon-button" aria-label="Dismiss message" onClick={() => setError(null)}><Icon name="close" /></button>}
      </div>}
    </section>
    <SidePanel tab={document ? sideTab : "references"} onTab={setSideTab} collapsed={panelCollapsed} onCollapsed={collapsePanel} layerCount={layers.layers.length} referenceCount={refs.references.length} showLayers={!!document}>
      {document && sideTab === "layers"
        ? <LayersPanel layers={layers.layers} activeId={activeLayer?.id ?? editLayer?.id ?? null} backgroundUrl={sourceUrl} uploading={layers.uploading} disabled={locked}
          canCopySelection={!!mask} onSelect={selectLayer} onAdd={files => void addLayerImages(files)} onCopySelection={() => void copySelectionToLayer()}
          onUpdate={layers.update} onPreview={layers.preview} onRemove={layers.remove} onDuplicate={layers.duplicate} onReorder={layers.reorder}
          onReset={resetLayer} onInvertMask={id => void invertLayerMask(id)} onBlend={id => void blendLayerIn(id)} onRemoveBackground={id => void selection.removeLayerBackground(id)}
          masking={layerMasking} onMask={toggleLayerMask} />
        : <ReferencePanel references={refs.references} newImage={!document} activeIds={refs.active.map(item => item.id)} pending={refs.pending} uploading={refs.uploading}
          disabled={locked || operation === "expand"} note={referenceNote}
          onUse={chooseReference} onAdd={files => void refs.add(files).then(id => { if (id !== null) chooseReference(id); })} onRemove={id => void refs.remove(id)} onIsolate={(id, isolated) => void refs.setIsolated(id, isolated)} />}
    </SidePanel>
    {panel === "settings" && <ControlDialog title="Generation settings" description="Choose how your image is created." onClose={() => setPanel(null)}><SettingsControls settings={settings} onChange={setSettings} models={models} task={task} disabled={locked} /></ControlDialog>}
    {panel === "expansion" && size && <ExpansionDialog size={size} expansion={expansion} onChange={setExpansion} disabled={locked} onClose={() => setPanel(null)} />}
    {panel === "shortcuts" && <ShortcutsDialog onClose={() => setPanel(null)} />}
    {libraryOpen && <LibraryDrawer documents={engine.documents} history={engine.history} onClose={() => setLibraryOpen(false)}
      onDocument={item => void perform(() => openSaved(item))}
      onGeneration={item => void perform(() => openGeneration(item))}
      onDeleteDocument={id => void perform(async () => { await deleteDocument(id); if (document?.id === id) newImage(); await refresh(); })}
      onDeleteGeneration={id => void perform(async () => { await deleteGeneration(id); await refresh(); })} />}
  </main>;
}
