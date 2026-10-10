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
}

/**
 * What to do with a selection, floating beside it like Photoshop's
 * contextual task bar, so the next step is one click away.
 */
export function SelectionActions(props: SelectionActionProps) {
  const { disabled } = props;
  return <div className="selection-actions-bar" role="toolbar" aria-label="Selection actions" onPointerDown={event => event.stopPropagation()}>
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
