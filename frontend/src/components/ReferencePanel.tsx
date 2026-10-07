import { DragEvent, useRef, useState } from "react";
import { ReferenceRecord, fileUrl } from "../api";
import { MAX_ACTIVE_REFERENCES } from "../hooks/useReferences";
import { Icon } from "./Icon";

interface Props {
  references: ReferenceRecord[];
  /** References used in the next edit, in prompt order. */
  activeIds: readonly number[];
  pending: ReadonlySet<number>;
  uploading: number;
  disabled: boolean;
  note: string | null;
  onUse: (id: number) => void;
  onAdd: (files: File[]) => void;
  onRemove: (id: number) => void;
  onIsolate: (id: number, isolated: boolean) => void;
}

export function ReferencePanel(props: Props) {
  const { references, activeIds, pending, uploading, disabled } = props;
  const full = activeIds.length >= MAX_ACTIVE_REFERENCES;
  const picker = useRef<HTMLInputElement>(null);
  const [dropping, setDropping] = useState(false);

  function drop(event: DragEvent<HTMLElement>) {
    event.preventDefault(); event.stopPropagation(); setDropping(false);
    if (disabled) return;
    const files = Array.from(event.dataTransfer.files);
    if (files.length) props.onAdd(files);
  }
  const dragHandlers = {
    onDragOver: (event: DragEvent<HTMLElement>) => { event.preventDefault(); event.stopPropagation(); if (!disabled) setDropping(true); },
    onDragLeave: (event: DragEvent<HTMLElement>) => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDropping(false); },
    onDrop: drop,
  };
  const input = <input ref={picker} className="visually-hidden" type="file" multiple accept="image/png,image/jpeg,image/webp"
    onChange={event => { const files = Array.from(event.target.files ?? []); if (files.length) props.onAdd(files); event.target.value = ""; }} />;

  return <div className={"reference-panel" + (dropping ? " dropping" : "")} {...dragHandlers}>
    <header>
      <div><small>{references.length > 1 ? `Pick up to ${MAX_ACTIVE_REFERENCES} to place in your image` : references.length ? "Pick it to place it in your image" : "People, pets or objects"}</small></div>
      <button className="icon-button" title="Add reference images" aria-label="Add reference images" disabled={disabled} onClick={() => picker.current?.click()}><Icon name="plus" /></button>
    </header>
    {input}
    {props.note && <p className="reference-note">{props.note}</p>}
    <div className="reference-list">
      {!references.length && !uploading && <button className="reference-empty" disabled={disabled} onClick={() => picker.current?.click()}>
        <Icon name="person" />
        <strong>Bring someone in</strong>
        <span>Add a photo of a person, pet or object. Pick it, select where it goes, then describe what it does.</span>
        <small>Drop images here or click to browse</small>
      </button>}
      {references.map(item => {
        const order = activeIds.indexOf(item.id) + 1, active = order > 0, busy = pending.has(item.id), isolated = !!item.subject_url;
        const title = active ? "Stop using this reference" : full ? `Up to ${MAX_ACTIVE_REFERENCES} references per edit. Take one out first.` : "Use this reference in your next edit";
        return <div key={item.id} className={"reference-card" + (active ? " active" : "") + (busy ? " busy" : "") + (!active && full ? " unavailable" : "")}>
          <button className="reference-use" aria-pressed={active} disabled={disabled || busy} title={title} onClick={() => props.onUse(item.id)}>
            <img crossOrigin="anonymous" src={fileUrl(item.subject_url ?? item.image_url)} alt="" loading="lazy" />
            {active && <span className="reference-badge">{activeIds.length > 1 ? `${order} · In use` : "In use"}</span>}
            {busy && <span className="reference-spinner" aria-label="Working" />}
          </button>
          <div className="reference-meta">
            <span title={item.name}>{item.name}</span>
            <button className="icon-button" aria-pressed={isolated} disabled={disabled || busy}
              title={isolated ? "Use the full image again" : "Isolate subject: remove the background"} aria-label={isolated ? "Use the full image" : "Isolate subject"}
              onClick={() => props.onIsolate(item.id, !isolated)}><Icon name="scissors" /></button>
            <button className="icon-button" disabled={disabled || busy} title="Remove reference" aria-label={`Remove ${item.name}`} onClick={() => props.onRemove(item.id)}><Icon name="trash" /></button>
          </div>
        </div>;
      })}
      {Array.from({ length: uploading }, (_, index) => <div key={"upload-" + index} className="reference-card uploading"><span className="reference-spinner" aria-label="Uploading" /></div>)}
      {references.length > 0 && <button className="reference-add" disabled={disabled} onClick={() => picker.current?.click()}><Icon name="plus" />Add more</button>}
    </div>
    {dropping && <div className="reference-drop">Drop to add references</div>}
  </div>;
}
