import { Operation, promptPlaceholder, submitLabel } from "../generation/operation";
import { Icon } from "./Icon";

interface Props {
  operation: Operation;
  onOperation: (operation: "replace" | "remove") => void;
  hasImage: boolean;
  hasSelection: boolean;
  locked: boolean;
  settingsLabel: string;
  prompt: string;
  onPrompt: (prompt: string) => void;
  hint: string;
  cancelled: boolean;
  jobActive: boolean;
  cancelling: boolean;
  canSubmit: boolean;
  onDeselect: () => void;
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
          {props.hasSelection && <button className="quiet-button" disabled={locked} onClick={props.onDeselect}>Deselect</button>}
        </> : operation === "expand" ? <>
          <span>Expand image</span>
          <button className="quiet-button" disabled={locked} onClick={props.onCanvasSize}><Icon name="expand" />Canvas size</button>
        </> : <span>New image</span>}
      </div>
      <button className="settings-trigger" aria-label="Generation settings" disabled={locked} onClick={props.onSettings}><span>{props.settingsLabel}</span><Icon name="settings" /></button>
    </div>
    <div className="prompt-row">
      <textarea aria-label="Describe your idea" rows={1} disabled={locked} placeholder={promptPlaceholder(operation, hasImage)} value={props.prompt}
        onChange={event => props.onPrompt(event.target.value)}
        onKeyDown={event => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); if (props.canSubmit) props.onSubmit(); } }} />
      {props.jobActive
        ? <button className="generate-button cancel-button" disabled={props.cancelling} onClick={props.onCancel}><Icon name="close" /><span>{props.cancelling ? "Cancelling…" : "Cancel"}</span></button>
        : <button className="primary generate-button" disabled={!props.canSubmit} onClick={props.onSubmit}><Icon name="spark" /><span>{submitLabel(operation)}</span><kbd>Ctrl ↵</kbd></button>}
    </div>
    <p className="command-hint" role={props.cancelled ? "status" : undefined}>{props.cancelled ? "Operation cancelled. Your image and selection are unchanged." : props.hint}</p>
  </section>;
}
