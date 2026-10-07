import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import {
  DocumentRecord, Expansion, GenerationRecord, JobResponse, JobStatus, ModelInfo, RevisionRecord, SelectionPoint, SystemInfo,
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
import { CancellationRequest, isActiveJob, useJobPolling } from "./hooks/useJobPolling";
import { MAX_ACTIVE_REFERENCES, useReferences } from "./hooks/useReferences";
import { useEngineWarmUp } from "./hooks/useEngineWarmUp";

interface JobContext { jobId: string; documentId: number | null; revisionId: number | null; operation: "inpaint" | "outpaint" | null }
const initialExpansion: Expansion = { left: 128, top: 0, right: 128, bottom: 0 };
const REFERENCES_COLLAPSED_KEY = "mosimage.referencesCollapsed";

export default function App() {
  const [settings, setSettings] = useState(defaultSettings);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [history, setHistory] = useState<GenerationRecord[]>([]);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [document, setDocument] = useState<DocumentRecord | null>(null);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [online, setOnline] = useState(false);
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
  const [referencesCollapsed, setReferencesCollapsed] = useState(() => { try { return localStorage.getItem(REFERENCES_COLLAPSED_KEY) === "1"; } catch { return false; } });
  const refs = useReferences(document?.id ?? null, setError);
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
  const cudaBlocked = !!system && (!system.torch_available || (system.preferred_device === "cuda" && !system.cuda_available && !system.allow_cpu_generation));

  useEffect(() => {
    const url = source ? URL.createObjectURL(source) : null;
    setSourceUrl(url);
    return () => { if (url) URL.revokeObjectURL(url); };
  }, [source]);

  async function refresh() {
    const [nextHistory, nextDocuments] = await Promise.all([listHistory(), listDocuments()]);
    setHistory(nextHistory); setDocuments(nextDocuments);
  }
  async function connect() {
    try {
      const [nextModels, nextSystem] = await Promise.all([listModels(), getSystemInfo(), refresh()]);
      setModels(nextModels); setSystem(nextSystem); setOnline(true); setError(null);
    } catch { setOnline(false); setError("The local engine is unavailable. Start MosImage with start.cmd, then reconnect."); }
  }
  useEffect(() => { void connect(); }, []);
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
      if (file) { event.preventDefault(); void perform(() => openFile(file)); }
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
  }, [locked, busy, reviewing, alternatives.length, selected, proposal, document, source, models, settings.model_name, operation]);

  useJobPolling({
    job, setJob, cancellation,
    onError: setError,
    onDisconnected: () => setOnline(false),
    onCompleted: async (next, isDisposed) => {
      if (next.mask_image_url) setMaskUrl(fileUrl(next.mask_image_url));
      if (next.generation_ids.length) {
        const submitted = context.current?.jobId === next.id ? context.current : null;
        const generated = await Promise.all(next.generation_ids.map(getGeneration));
        const revision = submitted?.documentId && submitted.operation
          ? await proposeAlternatives(submitted.documentId, next.generation_ids, submitted.operation, submitted.revisionId ?? undefined) : null;
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
    setMask(null); setMaskUrl(null); setPoints([]); setAlternatives([]); setProposal(null); setCompare(false);
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
    const record = direction === "undo" ? await undoDocument(document.id) : await redoDocument(document.id);
    await openSaved(record); await refresh();
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
  function collapseReferences(collapsed: boolean) {
    setReferencesCollapsed(collapsed);
    try { localStorage.setItem(REFERENCES_COLLAPSED_KEY, collapsed ? "1" : "0"); } catch { /* Preference only. */ }
  }
  function clearSelection() { setClearToken(value => value + 1); setMask(null); setMaskUrl(null); setPoints([]); }
  function queued(created: JobResponse, operationType: JobContext["operation"]) {
    cancellation.current = null;
    context.current = { jobId: created.id, documentId: document?.id ?? null, revisionId: document?.current_revision_id ?? null, operation: operationType };
    setJob({ ...created, progress: 0, error: null, generation_id: null, generation_ids: [] });
  }
  async function objectPoint(point: SelectionPoint) {
    if (!source || locked) return;
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
      let prompt = settings.prompt.trim();
      if (operation === "remove") {
        prompt = isKleinModel(settings.model_name)
          ? prompt || "Continue the surrounding background."
          : "Remove the selected object completely. Reconstruct the background naturally, matching the surrounding lighting, texture and perspective. " + prompt;
      }
      const parameters = { prompt, negative_prompt: settings.negative_prompt || undefined, model_name: settings.model_name, runtime_mode: settings.runtime_mode, seed: settings.seed ? Number(settings.seed) : undefined, steps: settings.steps, guidance: settings.guidance, count: settings.count };
      if (task === "text-to-image") queued(await generateImage({ ...parameters, width: settings.width, height: settings.height }), null);
      else if (task === "outpaint" && source && size && validExpansion(size, expansion)) queued(await outpaintImage(source, { ...parameters, ...expansion }), "outpaint");
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
  function exportImage() {
    if (!sourceUrl || !source) return;
    const anchor = window.document.createElement("a"); anchor.href = sourceUrl;
    const extension = source.type === "image/jpeg" ? ".jpg" : source.type === "image/webp" ? ".webp" : ".png";
    anchor.download = (document?.name ?? "MosImage").replace(/[<>:"/\\|?*]+/g, "-").replace(/\.[^.]+$/, "") + extension;
    anchor.click();
  }

  const expansionValid = !!size && validExpansion(size, expansion);
  const canSubmit = !locked && online && !cudaBlocked && (task === "outpaint"
    ? expansionValid && !!settings.prompt.trim()
    : (operation === "remove" || !!settings.prompt.trim()) && (task === "text-to-image" || !!mask) && !referenceNeedsQuality);
  const hint = referenceNeedsQuality ? "References need the Quality model. Choose Quality in generation settings."
    : operationHint(operation, !!source, !!mask, tool, references.map(item => item.name));
  const referenceNote = operation === "expand" ? "References are used for Replace edits. Leave Expand to place one."
    : operation === "remove" && refs.active.length ? "Switch to Replace to place the selected references." : null;

  return <main className={"focus-app" + (document ? " with-references" : "")}>
    <TopBar document={document} onRename={name => document && setDocument({ ...document, name })} online={online} system={system} warmUp={warmUp} locked={locked} canExport={!!source}
      onUndo={() => void perform(() => travel("undo"))} onRedo={() => void perform(() => travel("redo"))}
      onOpen={() => upload.current?.click()} onLibrary={() => setLibraryOpen(true)} onExport={exportImage} />
    <input ref={upload} className="visually-hidden" type="file" accept="image/png,image/jpeg,image/webp" onChange={onFile} />
    <ToolRail hasImage={!!source} tool={tool} expanding={operation === "expand"} locked={locked} onNew={newImage} onTool={chooseTool} onExpand={chooseExpansion} />
    <section className="focus-workspace" onDragOver={event => { event.preventDefault(); if (!locked) setDropActive(true); }} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDropActive(false); }} onDrop={drop}>
      <div className="canvas-area">
        {source && sourceUrl ? operation === "expand" && size
          ? <ExpansionCanvas imageUrl={sourceUrl} size={size} expansion={expansion} onChange={setExpansion} disabled={locked} reviewing={reviewing} previewUrl={previewUrl} previewSize={chosen?.width && chosen.height ? { width: chosen.width, height: chosen.height } : null} />
          : <EditorCanvas imageUrl={sourceUrl} previewImageUrl={previewUrl} reviewing={reviewing} imageName={document?.name} onMaskChange={setMask} tool={tool} disabled={locked} maskImportUrl={maskUrl} clearToken={clearToken}
            points={points} onObjectPoint={point => void objectPoint(point)} onRemovePoint={index => void removePoint(index)} onClearSelection={clearSelection}
            onImageSize={setSize} onError={setError} contextPadding={settings.context_padding} onContextPadding={context_padding => setSettings(current => ({ ...current, context_padding }))} onHelp={() => setPanel("shortcuts")} />
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
        {!online && <button onClick={() => void connect()}>Reconnect</button>}
        {error && <button className="icon-button" aria-label="Dismiss message" onClick={() => setError(null)}><Icon name="close" /></button>}
      </div>}
    </section>
    {document && <ReferencePanel references={refs.references} activeIds={refs.active.map(item => item.id)} pending={refs.pending} uploading={refs.uploading}
      disabled={locked || operation === "expand"} note={referenceNote} collapsed={referencesCollapsed} onCollapsed={collapseReferences}
      onUse={chooseReference} onAdd={files => void refs.add(files).then(id => { if (id !== null) chooseReference(id); })} onRemove={id => void refs.remove(id)} onIsolate={(id, isolated) => void refs.setIsolated(id, isolated)} />}
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
