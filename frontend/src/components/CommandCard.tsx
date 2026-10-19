import { RefObject, useLayoutEffect, useRef, useState } from "react";
import { ReferenceRole } from "../api";
import { EditScope, EnhanceLevel, FaceMode, Operation, SketchScope, promptPlaceholder, submitLabel } from "../generation/operation";
import { Icon } from "./Icon";
import { MOD } from "../platform";
import { promptHeight } from "./promptHeight";

export interface ActiveReference { id: number; name: string; imageUrl: string; role?: ReferenceRole }

const ROLE_BADGE: Record<ReferenceRole, string> = {
  subject: "Subject",
  clothing: "Outfit",
  pose: "Pose",
  style: "Style",
};

interface Props {
  promptRef?: RefObject<HTMLTextAreaElement | null>;
  operation: Operation;
  onOperation: (operation: Exclude<Operation, "expand">) => void;
  hasImage: boolean;
  locked: boolean;
  /** Offered for Edit once the document has layers. */
  editScope?: EditScope;
  onEditScope?: (scope: EditScope) => void;
  sketchScope: SketchScope;
  onSketchScope: (scope: SketchScope) => void;
  faceMode: FaceMode;
  onFaceMode: (mode: FaceMode) => void;
  enhanceLevel: EnhanceLevel;
  onEnhanceLevel: (level: EnhanceLevel) => void;
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
  const ownField = useRef<HTMLTextAreaElement>(null);
  const field = props.promptRef ?? ownField;
  const [focused, setFocused] = useState(false);
  // A disabled textarea may lose focus without a blur event, so locking collapses it too.
  const expanded = focused && !locked;
  // While focused the prompt grows upward over the canvas, so the layout below never shifts.
  useLayoutEffect(() => {
    const element = field.current;
    if (!element) return;
    element.style.height = "";
    if (expanded) element.style.height = `${promptHeight(element.scrollHeight, element.offsetHeight - element.clientHeight, element.offsetHeight, true)}px`;
    else element.scrollTop = 0;
  }, [field, expanded, props.prompt]);
  return <section className={"command-card" + (locked ? " locked" : "")} aria-label="Instruction">
    <div className="command-context">
      <div className="operation-controls">
        {hasImage && operation !== "expand" ? <>
          <div className="segmented">
            <button aria-pressed={operation === "generate"} disabled={locked} onClick={() => props.onOperation("generate")}>Generate</button>
            <button aria-pressed={operation === "edit"} disabled={locked} title="Change the whole picture with an instruction" onClick={() => props.onOperation("edit")}>Edit</button>
            <button aria-pressed={operation === "replace"} disabled={locked} onClick={() => props.onOperation("replace")}>Replace</button>
            <button aria-pressed={operation === "remove"} disabled={locked} onClick={() => props.onOperation("remove")}>Remove</button>
            <button aria-pressed={operation === "sketch"} disabled={locked} title="Paint rough shapes and let the AI render them · P" onClick={() => props.onOperation("sketch")}>Sketch</button>
            <button aria-pressed={operation === "faces"} disabled={locked} title="Sharpen blurry or distorted faces" onClick={() => props.onOperation("faces")}>Faces</button>
            <button aria-pressed={operation === "enhance"} disabled={locked} title="Add fine detail to the picture or the selection" onClick={() => props.onOperation("enhance")}>Enhance</button>
          </div>
          {operation === "faces" && <div className="segmented compact" aria-label="How to fix faces">
            <button aria-pressed={props.faceMode === "faithful"} disabled={locked} title="Sharpen each face from itself; expression and lighting stay" onClick={() => props.onFaceMode("faithful")}>Faithful</button>
            <button aria-pressed={props.faceMode === "redraw"} disabled={locked} title="Draw each face anew, for faces too blurry to sharpen; the expression can change" onClick={() => props.onFaceMode("redraw")}>Redraw</button>
          </div>}
          {operation === "enhance" && <div className="segmented compact" aria-label="How much detail">
            <button aria-pressed={props.enhanceLevel === "subtle"} disabled={locked} title="Believable texture; colours and shapes stay" onClick={() => props.onEnhanceLevel("subtle")}>Subtle</button>
            <button aria-pressed={props.enhanceLevel === "strong"} disabled={locked} title="More texture; small details may be reinvented" onClick={() => props.onEnhanceLevel("strong")}>Strong</button>
          </div>}
          {operation === "sketch" && <div className="segmented compact" aria-label="What to render">
            <button aria-pressed={props.sketchScope === "strokes"} disabled={locked} title="Render only the sketched area; the rest stays exactly as it is" onClick={() => props.onSketchScope("strokes")}>Only the sketch</button>
            <button aria-pressed={props.sketchScope === "whole"} disabled={locked} title="Redraw the whole picture around the sketch, for matching light and style" onClick={() => props.onSketchScope("whole")}>Whole picture</button>
          </div>}
          {operation === "edit" && props.onEditScope && <div className="segmented compact" aria-label="What to edit">
            <button aria-pressed={props.editScope === "visible"} disabled={locked} title="Edit the picture with every visible layer" onClick={() => props.onEditScope?.("visible")}>All visible</button>
            <button aria-pressed={props.editScope === "layer"} disabled={locked} title="Edit only the selected layer" onClick={() => props.onEditScope?.("layer")}>Current layer</button>
          </div>}
        </> : operation === "expand" ? <>
          <span>Expand image</span>
          <button className="quiet-button" disabled={locked} onClick={props.onCanvasSize}><Icon name="expand" />Canvas size</button>
        </> : <span>New image</span>}
      </div>
      <button className="settings-trigger" aria-label="Generation settings" disabled={locked} onClick={props.onSettings}><span>{props.settingsLabel}</span><Icon name="settings" /></button>
    </div>
    <div className="prompt-row">
      {props.references.length > 0 && <div className={"reference-chips" + (props.references.length > 1 ? " several" : "")}>
        {props.references.map(reference => <div key={reference.id} className="reference-chip" title={`Using ${reference.name} (${ROLE_BADGE[reference.role ?? "subject"]}) as reference`}>
          <img crossOrigin="anonymous" src={reference.imageUrl} alt="" />
          <span>{reference.name}</span>
          {reference.role && reference.role !== "subject" && <small className="reference-chip-role">{ROLE_BADGE[reference.role]}</small>}
          <button aria-label={`Stop using ${reference.name}`} title="Stop using this reference" disabled={locked} onClick={() => props.onRemoveReference(reference.id)}><Icon name="close" /></button>
        </div>)}
      </div>}
      <div className={"prompt-field" + (expanded ? " expanded" : "")}>
        <textarea ref={field} aria-label="Describe your idea" rows={1} disabled={locked} placeholder={promptPlaceholder(operation, hasImage, props.references.map(reference => reference.name))} value={props.prompt}
          onChange={event => props.onPrompt(event.target.value)} onFocus={() => setFocused(true)} onBlur={() => setFocused(false)}
          onKeyDown={event => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); if (props.canSubmit) props.onSubmit(); } }} />
      </div>
      {props.jobActive
        ? <button className="generate-button cancel-button" disabled={props.cancelling} onClick={props.onCancel}><Icon name="close" /><span>{props.cancelling ? "Cancelling…" : "Cancel"}</span></button>
        : <button className="primary generate-button" disabled={!props.canSubmit} onClick={props.onSubmit}><Icon name="spark" /><span>{submitLabel(operation)}</span><kbd>{MOD} ↵</kbd></button>}
    </div>
    <p className="command-hint" role={props.cancelled ? "status" : undefined}>{props.cancelled ? "Operation cancelled. Your image and selection are unchanged." : props.hint}</p>
  </section>;
}
