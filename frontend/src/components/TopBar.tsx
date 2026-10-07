import { DocumentRecord, JobStatus, SystemInfo } from "../api";
import { describeJob } from "../generation/progress";
import { isActiveJob } from "../hooks/useJobPolling";
import { Icon } from "./Icon";

interface Props {
  document: DocumentRecord | null;
  onRename: (name: string) => void;
  online: boolean;
  system: SystemInfo | null;
  /** Still trying to reach the engine, for example while the desktop app starts it. */
  connecting: boolean;
  warmUp: JobStatus | null;
  locked: boolean;
  canExport: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onOpen: () => void;
  onLibrary: () => void;
  onExport: () => void;
}

export function TopBar({ document, onRename, online, connecting, system, warmUp, locked, canExport, onUndo, onRedo, onOpen, onLibrary, onExport }: Props) {
  return <header className="topbar">
    <a className="brand" href="#" onClick={event => event.preventDefault()} aria-label="MosImage"><span className="brand-mark">m<span>•</span></span><strong>MosImage</strong></a>
    <div className="document-title">
      {document
        ? <><input aria-label="Document name" maxLength={160} value={document.name} onChange={event => onRename(event.target.value)} /><small>Saved locally</small></>
        : <strong>New image</strong>}
    </div>
    <div className="topbar-actions">
      <div className="document-actions">
        <button className="icon-button" aria-label="Undo edit" title="Undo edit · Ctrl + Z" disabled={locked || !document?.can_undo} onClick={onUndo}><Icon name="undo" /></button>
        <button className="icon-button" aria-label="Redo edit" title="Redo edit · Ctrl + Shift + Z" disabled={locked || !document?.can_redo} onClick={onRedo}><Icon name="redo" /></button>
        <button className="quiet-button" aria-label="Open image" title="Open image · Ctrl + O" disabled={locked} onClick={onOpen}><Icon name="upload" /><span>Open</span></button>
      </div>
      <EngineStatus online={online} connecting={connecting} system={system} warmUp={warmUp} />
      <button className="quiet-button" aria-label="Library" disabled={locked} onClick={onLibrary}><Icon name="library" /><span>Library</span></button>
      <button className="export-button" aria-label="Export image" title="Export the current image" disabled={!canExport || locked} onClick={onExport}><Icon name="download" /><span>Export</span></button>
    </div>
  </header>;
}

function EngineStatus({ online, connecting, system, warmUp }: { online: boolean; connecting: boolean; system: SystemInfo | null; warmUp: JobStatus | null }) {
  const device = system?.cuda_device ?? "Local engine";
  if (!online && connecting) return <span className="engine-status warming" role="status" title="Starting the local engine…"><i />Starting</span>;
  if (!online) return <span className="engine-status offline" title={device}><i />Offline</span>;
  if (warmUp && isActiveJob(warmUp)) {
    const text = describeJob(warmUp);
    return <span className="engine-status warming" role="status" title={`${text.title} ${text.detail}. You can start working; edits wait until it is ready.`}>
      <i />Warming up{text.percent !== null && <small>{text.percent}%</small>}
    </span>;
  }
  const failed = warmUp?.status === "failed";
  const title = failed ? `${device} · Warm-up skipped (${warmUp?.error}). Models load on first use.` : `${device} · Models ready`;
  return <span className={"engine-status online" + (failed ? " cold" : "")} title={title}><i />{failed ? "Local" : "Ready"}</span>;
}
