import { useEffect, useRef, useState } from "react";
import { DocumentRecord, GenerationRecord, fileUrl } from "../api";
import { Icon } from "./Icon";

interface Props {
  documents: DocumentRecord[];
  history: GenerationRecord[];
  onDocument: (document: DocumentRecord) => void;
  onGeneration: (generation: GenerationRecord) => void;
  onClose: () => void;
}

export function LibraryDrawer({ documents, history, onDocument, onGeneration, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [tab, setTab] = useState<"documents" | "history">("documents");
  const [query, setQuery] = useState("");
  useEffect(() => { dialog.current?.showModal(); }, []);
  function close() { dialog.current?.close(); onClose(); }
  const search = query.trim().toLocaleLowerCase();
  const visibleDocuments = documents.filter(item => item.name.toLocaleLowerCase().includes(search));
  const visibleHistory = history.filter(item => item.prompt.toLocaleLowerCase().includes(search));
  return <dialog ref={dialog} className="library-drawer" aria-label="Your library" onCancel={event => { event.preventDefault(); close(); }} onClick={event => { if (event.target === event.currentTarget) close(); }}>
    <div className="drawer-content">
      <header><div><span className="eyebrow">Everything stays on this computer</span><h2>Your library</h2></div><button className="icon-button" aria-label="Close library" onClick={close}><Icon name="close" /></button></header>
      <div className="segmented"><button aria-pressed={tab === "documents"} onClick={() => setTab("documents")}>Documents <small>{documents.length}</small></button><button aria-pressed={tab === "history"} onClick={() => setTab("history")}>Generations <small>{history.length}</small></button></div>
      <input className="library-search" type="search" placeholder="Search images…" aria-label="Search library" value={query} onChange={event => setQuery(event.target.value)} />
      <div className="library-grid">
        {tab === "documents" ? visibleDocuments.map(item => <button key={item.id} onClick={() => onDocument(item)}>{item.current_image_url && <img crossOrigin="anonymous" src={fileUrl(item.current_image_url)} alt="" loading="lazy" />}<strong>{item.name}</strong><small>{item.canvas_width} × {item.canvas_height}</small></button>) : visibleHistory.map(item => <button key={item.id} onClick={() => onGeneration(item)}><img crossOrigin="anonymous" src={fileUrl(item.output_image_url)} alt="" loading="lazy" /><strong>{item.prompt}</strong><small>{item.width} × {item.height}</small></button>)}
      </div>
      {!(tab === "documents" ? visibleDocuments.length : visibleHistory.length) && <p className="library-empty">{query ? "No images found." : "Your images will appear here as you create."}</p>}
    </div>
  </dialog>;
}
