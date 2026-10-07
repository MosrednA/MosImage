import { Operation, promptPlaceholder, submitLabel } from "../generation/operation";
import { Icon } from "./Icon";

export interface ActiveReference { id: number; name: string; imageUrl: string }

interface Props {
  operation: Operation;
  onOperation: (operation: "replace" | "remove") => void;
  hasImage: boolean;
  locked: boolean;
  settingsLabel: string;
  prompt: string;
  references: readonly ActiveReference[];
  onRemoveReference: (id: number) => void;
  onPrompt: (prompt: string) => void;
  hint: string;
  cancelled: boolean;
  jobActive: boolean;
  cancelling: boolean;
  canSubmit: boolean;
  onCanvasSize: () => void;
  onSettings: () => void;
  onSubmit: () => void;
  onCancel: () => void;
}

export function CommandCard(props: Props) {
  const { operation, hasImage, locked } = props;
  return <section className={"command-card" + (locked ? " locked" : "")} aria-label="Instruction">
    <div className="command-context">
      <div className="operation-controls">
        {hasImage && operation !== "expand" ? <>
          <div className="segmented">
            <button aria-pressed={operation === "replace"} disabled={locked} onClick={() => props.onOperation("replace")}>Replace</button>
            <button aria-pressed={operation === "remove"} disabled={locked} onClick={() => props.onOperation("remove")}>Remove</button>
          </div>
        </> : operation === "expand" ? <>
          <span>Expand image</span>
          <button className="quiet-button" disabled={locked} onClick={props.onCanvasSize}><Icon name="expand" />Canvas size</button>
        </> : <span>New image</span>}
      </div>
      <button className="settings-trigger" aria-label="Generation settings" disabled={locked} onClick={props.onSettings}><span>{props.settingsLabel}</span><Icon name="settings" /></button>
    </div>
    <div className="prompt-row">
      {props.references.length > 0 && <div className={"reference-chips" + (props.references.length > 1 ? " several" : "")}>
        {props.references.map(reference => <div key={reference.id} className="reference-chip" title={`Using ${reference.name} as reference`}>
          <img crossOrigin="anonymous" src={reference.imageUrl} alt="" /><span>{reference.name}</span>
          <button aria-label={`Stop using ${reference.name}`} title="Stop using this reference" disabled={locked} onClick={() => props.onRemoveReference(reference.id)}><Icon name="close" /></button>
        </div>)}
      </div>}
      <textarea aria-label="Describe your idea" rows={1} disabled={locked} placeholder={promptPlaceholder(operation, hasImage, props.references.map(reference => reference.name))} value={props.prompt}
        onChange={event => props.onPrompt(event.target.value)}
        onKeyDown={event => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); if (props.canSubmit) props.onSubmit(); } }} />
      {props.jobActive
        ? <button className="generate-button cancel-button" disabled={props.cancelling} onClick={props.onCancel}><Icon name="close" /><span>{props.cancelling ? "Cancelling…" : "Cancel"}</span></button>
        : <button className="primary generate-button" disabled={!props.canSubmit} onClick={props.onSubmit}><Icon name="spark" /><span>{submitLabel(operation)}</span><kbd>Ctrl ↵</kbd></button>}
    </div>
    <p className="command-hint" role={props.cancelled ? "status" : undefined}>{props.cancelled ? "Operation cancelled. Your image and selection are unchanged." : props.hint}</p>
  </section>;
}
