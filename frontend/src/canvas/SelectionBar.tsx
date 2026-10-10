import { Icon } from "../components/Icon";
import { EXPAND_RANGE, SOFTEN_RANGE } from "../masks/maskRefine";

interface Props {
  disabled: boolean;
  expand: number;
  onExpand: (value: number) => void;
  soften: number;
  softenAuto: boolean;
  onSoften: (value: number | null) => void;
  contextPadding: number;
  onContextPadding: (value: number) => void;
  showContext: boolean;
  onToggleContext: () => void;
  maskVisible: boolean;
  onToggleMask: () => void;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onInvert: () => void;
  onClear: () => void;
}

/**
 * Edge controls for the current selection. Every value previews live on the
 * canvas: the blue tint is exactly the mask that will be regenerated.
 */
export function SelectionBar(props: Props) {
  const { disabled } = props;
  return <div className="selection-bar" role="toolbar" aria-label="Selection edges">
    <div className="edge-slider" title="Grow or shrink the selection edge">
      <span>Expand</span>
      <input aria-label="Expand selection" type="range" min={EXPAND_RANGE.min} max={EXPAND_RANGE.max} value={props.expand} disabled={disabled} onChange={event => props.onExpand(Number(event.target.value))} onDoubleClick={() => props.onExpand(0)} />
      <output>{props.expand > 0 ? "+" : ""}{props.expand}px</output>
    </div>
    <div className="edge-slider" title="Width of the soft transition into the original image">
      <span>Soften</span>
      <input aria-label="Soften edge" type="range" min={SOFTEN_RANGE.min} max={SOFTEN_RANGE.max} value={props.soften} disabled={disabled} onChange={event => props.onSoften(Number(event.target.value))} />
      <output>{props.soften}px</output>
      <button type="button" className={"chip" + (props.softenAuto ? " selected" : "")} aria-pressed={props.softenAuto} disabled={disabled} title="Scale the transition with the image" onClick={() => props.onSoften(null)}>Auto</button>
    </div>
    <div className="edge-slider" title="Surroundings the model sees; they stay unchanged">
      <button type="button" className={"chip" + (props.showContext ? " selected" : "")} aria-pressed={props.showContext} onClick={props.onToggleContext}>Context</button>
      <input aria-label="Surrounding context" type="range" min="32" max="1024" step="32" value={props.contextPadding} disabled={disabled} onChange={event => props.onContextPadding(Number(event.target.value))} />
      <output>{props.contextPadding}px</output>
    </div>
    <div className="selection-actions">
      <button className="icon-button" aria-pressed={!props.maskVisible} aria-label={props.maskVisible ? "Hide selection overlay" : "Show selection overlay"} title="Show or hide the selection · Q" onClick={props.onToggleMask}><Icon name={props.maskVisible ? "eye" : "eyeOff"} /></button>
      <button className="icon-button" aria-label="Invert selection" title="Invert selection · Ctrl + Shift + I" disabled={disabled} onClick={props.onInvert}><Icon name="invert" /></button>
      <button className="icon-button" aria-label="Undo selection change" title="Undo selection change · Ctrl + Z" disabled={disabled || !props.canUndo} onClick={props.onUndo}><Icon name="undo" /></button>
      <button className="icon-button" aria-label="Redo selection change" title="Redo selection change · Ctrl + Shift + Z" disabled={disabled || !props.canRedo} onClick={props.onRedo}><Icon name="redo" /></button>
      <button className="quiet-button" disabled={disabled} title="Clear the selection · Esc" onClick={props.onClear}>Clear</button>
    </div>
  </div>;
}
