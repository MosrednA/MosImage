import { ReactNode, useEffect, useRef } from "react";
import { Icon } from "./Icon";

interface Props { title: string; description: string; onClose: () => void; children: ReactNode }

export function ControlDialog({ title, description, onClose, children }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => { dialog.current?.showModal(); }, []);
  function close() { dialog.current?.close(); onClose(); }
  return <dialog ref={dialog} className="control-dialog" aria-label={title} onCancel={event => { event.preventDefault(); close(); }} onClick={event => { if (event.target === event.currentTarget) close(); }}>
    <div className="control-content">
      <header><div><h2>{title}</h2><p>{description}</p></div><button className="icon-button" aria-label="Close panel" onClick={close}><Icon name="close" /></button></header>
      <div className="control-body">{children}</div>
      <footer><button className="primary" onClick={close}>Done</button></footer>
    </div>
  </dialog>;
}
