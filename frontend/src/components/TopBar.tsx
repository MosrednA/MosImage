import { DocumentRecord, SystemInfo } from "../api";
import { Icon } from "./Icon";

interface Props {
  document: DocumentRecord | null;
  onRename: (name: string) => void;
  online: boolean;
  system: SystemInfo | null;
  locked: boolean;
  canExport: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onOpen: () => void;
  onLibrary: () => void;
  onExport: () => void;
}

export function TopBar({ document, onRename, online, system, locked, canExport, onUndo, onRedo, onOpen, onLibrary, onExport }: Props) {
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
      <span className={"engine-status " + (online ? "online" : "offline")} title={system?.cuda_device ?? "Local engine"}><i />{online ? "Local" : "Offline"}</span>
      <button className="quiet-button" aria-label="Library" disabled={locked} onClick={onLibrary}><Icon name="library" /><span>Library</span></button>
      <button className="export-button" aria-label="Export image" title="Export the current image" disabled={!canExport || locked} onClick={onExport}><Icon name="download" /><span>Export</span></button>
    </div>
  </header>;
}
