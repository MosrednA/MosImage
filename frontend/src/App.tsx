import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import {
  ApiError, DocumentRecord, Expansion, GenerationRecord, ImageTask, JobResponse, JobStatus, ModelInfo, RevisionRecord, SelectionPoint, SystemInfo,
  acceptAlternative, cancelJob, createDocument, fileUrl, generateImage, getDocument, getGeneration, getJob, getSystemInfo,
  importGeneration, inpaintImage, listDocuments, listHistory, listModels, outpaintImage, proposeAlternatives, redoDocument,
  selectObject, undoDocument, updateDocument,
} from "./api";
import { CanvasTool, EditorCanvas } from "./canvas/EditorCanvas";
import { ImageSize } from "./canvas/coordinates";
import { ExpansionCanvas } from "./canvas/ExpansionCanvas";
import { expansionForRatio, expansionSize, validExpansion } from "./canvas/expansion";
import { GENERATION_PRESETS } from "./generation/presets";
import { SettingsControls, defaultSettings } from "./components/EditorSettings";
import { Icon } from "./components/Icon";
import { ControlDialog } from "./components/ControlDialog";
import { LibraryDrawer } from "./components/LibraryDrawer";

type Operation = "generate" | "replace" | "remove" | "expand";
interface JobContext { documentId: number | null; revisionId: number | null; operation: "inpaint" | "outpaint" | null }
const initialExpansion: Expansion = { left: 128, top: 0, right: 128, bottom: 0 };

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
  const [panel, setPanel] = useState<"settings" | "expansion" | null>(null);
  const upload = useRef<HTMLInputElement>(null);
  const context = useRef<JobContext | null>(null);
  const cancellation = useRef<{ id: string; requestedAt: number } | null>(null);
  const transaction = useRef(false);
  const jobActive = job?.status === "queued" || job?.status === "running" || job?.status === "cancelling";
  const cancelling = job?.status === "cancelling";
  const busy = pending || jobActive;
  const locked = busy || alternatives.length > 0;
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  useEffect(() => {
    const url = source ? URL.createObjectURL(source) : null;
    setSourceUrl(url);
    return () => { if (url) URL.revokeObjectURL(url); };
  }, [source]);
  const chosen = alternatives[selected];
  const previewUrl = chosen && !compare ? fileUrl(proposal?.alternatives[selected]?.output_image_url ?? chosen.output_image_url) : null;
  const task: ImageTask = operation === "generate" ? "text-to-image" : operation === "expand" ? "outpaint" : "inpaint";
  const cudaBlocked = !!system && (!system.torch_available || (system.preferred_device === "cuda" && !system.cuda_available && !system.allow_cpu_generation));

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
      if (typing || locked) return;
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "o") { event.preventDefault(); upload.current?.click(); }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z" && document) {
        event.preventDefault();
        if (event.shiftKey ? document.can_redo : document.can_undo) void perform(() => travel(event.shiftKey ? "redo" : "undo"));
      }
      if (!event.ctrlKey && !event.metaKey && source) {
        if (event.key.toLowerCase() === "b") chooseTool("brush");
        if (event.key.toLowerCase() === "o") chooseTool("object");
        if (event.key.toLowerCase() === "v") chooseTool("move");
        if (event.key.toLowerCase() === "e") chooseExpansion();
      }
    };
    window.addEventListener("paste", paste); window.addEventListener("keydown", keys);
    return () => { window.removeEventListener("paste", paste); window.removeEventListener("keydown", keys); };
  }, [locked, document, source, models, settings.model_name]);

  useEffect(() => {
    if (!job) return;
    const id = job.id, submitted = context.current;
    let disposed = false, timer = 0;
    async function poll() {
      try {
        const next = await getJob(id);
        if (disposed) return;
        // A poll already in flight can return an older running state after Cancel.
        if (cancellation.current?.id === id && (next.status === "queued" || next.status === "running")) next.status = "cancelling";
        if (next.status === "completed") {
          if (next.mask_image_url) setMaskUrl(fileUrl(next.mask_image_url));
          if (next.generation_ids.length) {
            const generated = await Promise.all(next.generation_ids.map(getGeneration));
            const revision = submitted?.documentId && submitted.operation
              ? await proposeAlternatives(submitted.documentId, next.generation_ids, submitted.operation, submitted.revisionId ?? undefined) : null;
            if (disposed) return;
            setAlternatives(generated); setProposal(revision); setSelected(0); setCompare(false);
            await refresh();
          }
          setModels(await listModels());
        }
        if (disposed) return;
        setJob(next);
        if (next.status === "cancelled" || next.status === "completed" || next.status === "failed") cancellation.current = null;
        if (next.status === "failed") setError(next.error ?? "The operation failed.");
        if (next.status === "queued" || next.status === "running" || next.status === "cancelling") timer = window.setTimeout(poll, 800);
      } catch (err) {
        if (disposed) return;
        if (cancellation.current?.id === id) {
          // A supervised hard cancel restarts the worker and removes its in-memory jobs.
          if (err instanceof ApiError && err.status === 404) {
            cancellation.current = null;
            setJob(current => current?.id === id ? { ...current, status: "cancelled" } : current);
            return;
          }
          if (Date.now() - cancellation.current.requestedAt < 30_000) {
            timer = window.setTimeout(poll, 800);
            return;
          }
          setOnline(false);
          setError("Cancellation was requested, but the local engine has not reconnected. Reconnect to continue.");
        } else setError(message(err));
        cancellation.current = null;
        setJob(current => current?.id === id ? { ...current, status: "failed" } : current);
      }
    }
    timer = window.setTimeout(poll, 400);
    return () => { disposed = true; window.clearTimeout(timer); };
  }, [job?.id]);

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
  function clearSelection() { setClearToken(value => value + 1); setMask(null); setMaskUrl(null); setPoints([]); }
  function queued(created: JobResponse, operationType: JobContext["operation"]) {
    cancellation.current = null;
    context.current = { documentId: document?.id ?? null, revisionId: document?.current_revision_id ?? null, operation: operationType };
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
  async function submit() {
    if (locked || !online || cudaBlocked) return;
    await perform(async () => {
      let prompt = settings.prompt.trim();
      if (operation === "remove") {
        prompt = settings.model_name === "flux2-klein-4b"
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
        form.set("blend_feather", String(settings.blend_feather));
        form.set("context_padding", String(settings.context_padding));
        form.set("scene_context", settings.scene_context);
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
  const expansionValid = size && validExpansion(size, expansion);
  const canSubmit = !locked && online && !cudaBlocked && (task === "outpaint" ? expansionValid && settings.prompt.trim() : (operation === "remove" || settings.prompt.trim()) && (task === "text-to-image" || mask));
  const selectionHint = operation === "remove"
    ? "Include the object, its shadow and any glow. Grow the selection if edges remain."
    : "Describe the change. Leave room in your selection for the full subject and its shadow.";
  const hint = task === "outpaint" ? "Describe the scene or new content. Your original image stays intact."
    : source ? mask ? selectionHint : tool === "object" ? "Click an object. Alt + click to exclude parts." : "Select the area you want to change."
    : "Describe a new image, or open one to edit.";

  return <main className="focus-app">
    <header className="topbar">
      <a className="brand" href="#" onClick={event => event.preventDefault()} aria-label="MosImage"><span className="brand-mark">m<span>•</span></span><strong>MosImage</strong></a>
      <div className="document-title">{document ? <><input aria-label="Document name" maxLength={160} value={document.name} onChange={event => setDocument({ ...document, name: event.target.value })} /><small>Saved locally</small></> : <strong>New image</strong>}</div>
      <div className="topbar-actions">
        <div className="document-actions">
          <button className="icon-button" aria-label="Undo edit" title="Undo edit · Ctrl + Z" disabled={locked || !document?.can_undo} onClick={() => void perform(() => travel("undo"))}><Icon name="undo" /></button>
          <button className="icon-button" aria-label="Redo edit" title="Redo edit · Ctrl + Shift + Z" disabled={locked || !document?.can_redo} onClick={() => void perform(() => travel("redo"))}><Icon name="redo" /></button>
          <button className="quiet-button" aria-label="Open image" title="Open image · Ctrl + O" disabled={locked} onClick={() => upload.current?.click()}><Icon name="upload" /><span>Open</span></button>
        </div>
        <span className={"engine-status " + (online ? "online" : "offline")} title={system?.cuda_device ?? "Local engine"}><i />{online ? "Local" : "Offline"}</span>
        <button className="quiet-button" aria-label="Library" disabled={locked} onClick={() => setLibraryOpen(true)}><Icon name="library" /><span>Library</span></button>
        <button className="export-button" aria-label="Export image" title="Export the current image" disabled={!source || locked} onClick={exportImage}><Icon name="download" /><span>Export</span></button>
      </div>
    </header>
    <input ref={upload} className="visually-hidden" type="file" accept="image/png,image/jpeg,image/webp" onChange={onFile} />
    <nav className="tool-rail" aria-label="Tools">
      <button className={!source ? "active" : ""} title="New image" disabled={locked} onClick={newImage}><Icon name="spark" /><span>New</span></button>
      <div className="rail-divider" />
      <button className={source && tool === "object" && operation !== "expand" ? "active" : ""} title="Select object (O)" disabled={!source || locked} onClick={() => chooseTool("object")}><Icon name="object" /><span>Object</span></button>
      <button className={source && ["brush", "eraser", "rectangle", "lasso"].includes(tool) && operation !== "expand" ? "active" : ""} title="Brush (B)" disabled={!source || locked} onClick={() => chooseTool("brush")}><Icon name="brush" /><span>Brush</span></button>
      <button className={operation === "expand" ? "active" : ""} title="Expand image (E)" disabled={!source || locked} onClick={chooseExpansion}><Icon name="expand" /><span>Expand</span></button>
      <button className={tool === "move" && operation !== "expand" ? "active" : ""} title="Pan (V)" disabled={!source || locked} onClick={() => chooseTool("move")}><Icon name="move" /><span>Pan</span></button>
      <div className="rail-footer"><span>LOCAL<br />STUDIO</span></div>
    </nav>
    <section className="focus-workspace" onDragOver={event => { event.preventDefault(); if (!locked) setDropActive(true); }} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDropActive(false); }} onDrop={drop}>
      <div className="canvas-area">
        {source && sourceUrl ? operation === "expand" && size ? <ExpansionCanvas imageUrl={sourceUrl} size={size} expansion={expansion} onChange={setExpansion} disabled={locked} reviewing={!!alternatives.length} previewUrl={previewUrl} previewSize={chosen?.width && chosen.height ? { width: chosen.width, height: chosen.height } : null} />
          : <EditorCanvas imageUrl={sourceUrl} previewImageUrl={previewUrl} reviewing={!!alternatives.length} imageName={document?.name} onMaskChange={setMask} tool={tool} onToolChange={chooseTool} disabled={locked} maskImportUrl={maskUrl} clearToken={clearToken} points={points} onObjectPoint={point => void objectPoint(point)} onImageSize={setSize} onError={setError} contextPadding={settings.context_padding} />
          : previewUrl ? <div className="generated-preview"><img crossOrigin="anonymous" src={previewUrl} alt={chosen.prompt} /><span>{chosen.width} × {chosen.height}</span></div>
          : <div className="welcome-canvas"><div className="welcome-art" aria-hidden="true"><div /><div /><div /><Icon name="spark" /></div><span className="eyebrow">Your next idea starts here</span><h1>Make something<br /><em>beautiful.</em></h1><p>A new image from a few words.<br />Or give your own image a fresh direction.</p><button disabled={locked} onClick={() => upload.current?.click()}><Icon name="upload" />Open an image</button><small>Drop an image here · or paste with Ctrl + V</small></div>}
        {busy && <div className="job-overlay" role="status"><span className="spinner" /><div><strong>{job ? jobLabel(job) : "Opening image…"}</strong><small>{cancelling ? "Stopping the operation and releasing memory…" : job?.kind === "select" ? "The selection model downloads on first use" : (job?.progress ?? 0) <= .05 ? "The first run may take longer while the model loads" : "Your preview will appear as soon as it is ready"}</small></div>{job && !cancelling && <output>{Math.round(job.progress * 100)}%</output>}</div>}
        {dropActive && <div className="drop-overlay">Drop to open your image</div>}
      </div>
      {!!alternatives.length ? <section className="proposal-tray" aria-label="Proposals"><div className="proposal-intro"><strong>Choose a proposal</strong><span>{selected + 1} of {alternatives.length}</span></div><div className="proposal-thumbnails">{alternatives.map((item, index) => <button key={item.id} className={index === selected ? "selected" : ""} aria-pressed={index === selected} aria-label={"Proposal " + (index + 1)} onClick={() => { setSelected(index); setCompare(false); }}><img crossOrigin="anonymous" src={fileUrl(proposal?.alternatives[index]?.output_image_url ?? item.output_image_url)} alt="" /><span>{index + 1}</span></button>)}</div><div className="proposal-actions">{source && <button aria-pressed={compare} onClick={() => setCompare(value => !value)}>{compare ? "Show proposal" : "View original"}</button>}<button disabled={busy} onClick={() => { setAlternatives([]); setProposal(null); setCompare(false); }}>Discard</button><button className="primary" disabled={busy} onClick={() => void apply()}>Apply</button></div></section>
      : <section className={"command-card" + (locked ? " locked" : "")} aria-label="Instruction">
        <div className="command-context">
          <div className="operation-controls">{source && operation !== "expand" ? <><div className="segmented"><button aria-pressed={operation === "replace"} disabled={locked} onClick={() => setOperation("replace")}>Replace</button><button aria-pressed={operation === "remove"} disabled={locked} onClick={() => setOperation("remove")}>Remove</button></div>{(mask || points.length > 0) && <button className="quiet-button" disabled={locked} onClick={clearSelection}>Deselect</button>}</> : operation === "expand" ? <><span>Expand image</span><button className="quiet-button" disabled={locked} onClick={() => setPanel("expansion")}><Icon name="expand" />Canvas size</button></> : <span>New image</span>}</div>
          <button className="settings-trigger" aria-label="Generation settings" disabled={locked} onClick={() => setPanel("settings")}><span>{settings.model_name === "flux1-fill-dev" ? "Fill" : task === "outpaint" ? "Standard" : settings.preset === "fast" ? "Fast" : "Quality"} · {settings.count} {settings.count === 1 ? "proposal" : "proposals"}{task === "text-to-image" ? " · " + settings.width + " × " + settings.height : ""}</span><Icon name="settings" /></button>
        </div>
        <div className="prompt-row"><textarea aria-label="Describe your idea" rows={1} disabled={locked} placeholder={operation === "remove" ? "Optional: describe the background to restore…" : operation === "expand" ? "Describe the scene or what to add beyond the edges…" : source ? "What would you like to see in your selection?" : "Describe the image you want to create…"} value={settings.prompt} onChange={event => setSettings({ ...settings, prompt: event.target.value })} onKeyDown={event => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); if (canSubmit) void submit(); } }} />{jobActive ? <button className="generate-button cancel-button" disabled={cancelling} onClick={() => void cancel()}><Icon name="close" /><span>{cancelling ? "Cancelling…" : "Cancel"}</span></button> : <button className="primary generate-button" disabled={!canSubmit} onClick={() => void submit()}><Icon name="spark" /><span>{operation === "generate" ? "Generate" : operation === "remove" ? "Remove" : operation === "expand" ? "Expand" : "Generate"}</span><kbd>Ctrl ↵</kbd></button>}</div>
        <p className="command-hint" role={job?.status === "cancelled" ? "status" : undefined}>{job?.status === "cancelled" ? "Operation cancelled. Your image and selection are unchanged." : hint}</p>
      </section>}
      {(cudaBlocked || error) && <div className="error-notice" role="alert"><span>{error ?? system?.setup_hint ?? "CUDA is unavailable. Check the local PyTorch installation."}</span>{!online && <button onClick={() => void connect()}>Reconnect</button>}{error && <button className="icon-button" aria-label="Dismiss message" onClick={() => setError(null)}><Icon name="close" /></button>}</div>}
    </section>
    {panel === "settings" && <ControlDialog title="Generation settings" description="Choose how your image is created." onClose={() => setPanel(null)}><SettingsControls settings={settings} onChange={setSettings} models={models} task={task} disabled={locked} /></ControlDialog>}
    {panel === "expansion" && size && <ControlDialog title="Canvas size" description="Choose a format or add space on any side. You can also drag the canvas edges." onClose={() => setPanel(null)}>
      <fieldset disabled={locked} className="expansion-options"><div className="expansion-presets">{[["Wide 16:9", 16 / 9], ["Square", 1], ["Portrait 4:5", .8]].map(([label, ratio]) => <button key={label} onClick={() => setExpansion(expansionForRatio(size, Number(ratio)))}>{label}</button>)}<button onClick={() => setExpansion({ left: 128, top: 128, right: 128, bottom: 128 })}>All sides +128</button></div><div className="margin-fields">{(["left", "top", "right", "bottom"] as const).map((side, index) => <label key={side}>{["Left", "Top", "Right", "Bottom"][index]}<input type="number" min="0" max="2048" step="32" value={expansion[side]} onChange={event => setExpansion({ ...expansion, [side]: Number(event.target.value) })} /></label>)}</div><div className="canvas-size-result">{expansionSize(size, expansion).width} × {expansionSize(size, expansion).height} px <small>Original pixels stay intact.</small></div>{!expansionValid && <p className="field-error">Add space and keep the canvas within 4096 × 4096 pixels.</p>}</fieldset>
    </ControlDialog>}
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
function jobLabel(job: JobStatus): string {
  if (job.status === "cancelling") return "Cancelling…";
  if (job.status === "queued") return "Waiting for the local engine…";
  if (job.kind === "select") return "Selecting object…";
  return job.progress <= .05 ? "Loading model…" : job.kind === "outpaint" ? "Expanding your image…" : "Creating your proposals…";
}
