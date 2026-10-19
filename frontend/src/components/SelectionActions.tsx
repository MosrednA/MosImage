import { BarGrip } from "../canvas/useBarSpot";
import { Icon } from "./Icon";
import { MOD } from "../platform";

export interface SelectionActionProps {
  disabled: boolean;
  /** Puts the cursor in the prompt to fill the selection. */
  onFill: () => void;
  onRemove: () => void;
  onFaces: () => void;
  onEnhance: () => void;
  onToLayer: () => void;
  /** The layer last worked on, which the selection can become the mask of. */
  maskTarget: string | null;
  onMaskLayer: () => void;
  onInvert: () => void;
  onDeselect: () => void;
  grip: BarGrip;
}

/**
 * What to do with a selection, floating beside it like Photoshop's
 * contextual task bar, so the next step is one click away. The grip moves it
 * out of the way.
 */
export function SelectionActions(props: SelectionActionProps) {
  const { disabled, grip } = props;
  return <div className={"selection-actions-bar" + (grip.dragging ? " dragging" : "")} role="toolbar" aria-label="Selection actions" onPointerDown={event => event.stopPropagation()}>
    <span className="selection-actions-grip" aria-hidden="true" onPointerDown={grip.onPointerDown} onDoubleClick={grip.onReset}
      title={grip.moved ? "Drag to move the bar · double-click to put it back by the selection" : "Drag to move the bar"}><Icon name="grip" /></span>
    <button className="primary" disabled={disabled} title="Describe what should appear in the selection · Shift + Backspace" onClick={props.onFill}><Icon name="spark" />Generate</button>
    <button disabled={disabled} title="Remove what is selected and fill in the background" onClick={props.onRemove}><Icon name="eraser" />Remove</button>
    <button disabled={disabled} title="Sharpen the faces inside the selection" onClick={props.onFaces}><Icon name="face" />Faces</button>
    <button disabled={disabled} title="Add fine detail inside the selection" onClick={props.onEnhance}><Icon name="enhance" />Enhance</button>
    <span className="selection-actions-divider" />
    <button disabled={disabled} title={`Copy the selection to a new layer · ${MOD} + J`} onClick={props.onToLayer}><Icon name="layers" />To layer</button>
    {props.maskTarget && <button disabled={disabled} title={`Show only the part of “${props.maskTarget}” inside the selection`} onClick={props.onMaskLayer}>
      <Icon name="scissors" /><span className="selection-actions-target">Mask “{props.maskTarget}”</span>
    </button>}
    <span className="selection-actions-divider" />
    <button className="icon-button" disabled={disabled} aria-label="Invert selection" title={`Invert selection · ${MOD} + Shift + I`} onClick={props.onInvert}><Icon name="invert" /></button>
    <button className="icon-button" disabled={disabled} aria-label="Deselect" title={`Deselect · ${MOD} + D`} onClick={props.onDeselect}><Icon name="close" /></button>
  </div>;
}
