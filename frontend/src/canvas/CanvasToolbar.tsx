import { CanvasTool, MASK_TOOLS, toolLabel } from "./tools";
import { MaskTransform } from "./useMaskLayer";

interface Props {
  tool: CanvasTool;
  onToolChange: (tool: CanvasTool) => void;
  reviewing: boolean;
  disabled: boolean;
  hasImage: boolean;
  hasSelection: boolean;
  showContext: boolean;
  onToggleContext: () => void;
  brushSize: number;
  onBrushSize: (size: number) => void;
  edgeSize: number;
  onEdgeSize: (size: number) => void;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onTransform: (transform: MaskTransform) => void;
  onClear: () => void;
  zoom: number;
  fitToStage: boolean;
  atActualSize: boolean;
  onZoom: (zoom: number) => void;
  onFit: () => void;
}

export function CanvasToolbar(props: Props) {
  const { tool, reviewing, disabled, hasSelection } = props;
  return <div className="editor-toolbar">
    {!reviewing && hasSelection && <button aria-pressed={props.showContext} className={props.showContext ? "selected" : ""} title="Show the protected context sent with your selection" onClick={props.onToggleContext}>Context</button>}
    {reviewing ? <span className="canvas-context-label">Review your proposal</span>
      : tool === "object" || tool === "move" ? <span className="canvas-context-label">{tool === "object" ? "Object selection" : "Pan image"}</span>
        : <div className="tool-group" aria-label="Selection tools">
          {MASK_TOOLS.map(candidate => <button key={candidate} className={tool === candidate ? "selected" : ""} disabled={disabled} onClick={() => props.onToolChange(candidate)}>{toolLabel(candidate)}</button>)}
        </div>}
    {!reviewing && (tool === "brush" || tool === "eraser") && <label className="brush-control">
      <span>Brush</span>
      <input type="range" min="8" max="256" value={props.brushSize} onChange={event => props.onBrushSize(Number(event.target.value))} />
      <output>{props.brushSize}px</output>
    </label>}
    {!reviewing && hasSelection && <details className="selection-details"><summary>Refine selection</summary><div className="selection-popover">
      <label className="edge-control">
        <span>Edge</span>
        <input type="range" min="1" max="64" value={props.edgeSize} onChange={event => props.onEdgeSize(Number(event.target.value))} />
        <output>{props.edgeSize}px</output>
      </label>
      <div className="tool-group mask-actions">
        <button disabled={disabled || !props.canUndo} onClick={props.onUndo}>Undo mask</button>
        <button disabled={disabled || !props.canRedo} onClick={props.onRedo}>Redo mask</button>
        <button disabled={disabled} onClick={() => props.onTransform("grow")}>Grow</button>
        <button disabled={disabled} onClick={() => props.onTransform("shrink")}>Shrink</button>
        <button disabled={disabled} onClick={() => props.onTransform("feather")}>Feather</button>
        <button disabled={disabled || !props.hasImage} onClick={() => props.onTransform("invert")}>Invert</button>
        <button disabled={disabled} onClick={props.onClear}>Clear</button>
      </div>
    </div></details>}
    <div className="zoom-controls">
      <button aria-label="Zoom out" onClick={() => props.onZoom(props.zoom - 0.1)}>−</button>
      <button className={props.fitToStage ? "selected" : ""} onClick={props.onFit}>Fit</button>
      <button className={props.atActualSize ? "selected" : ""} onClick={() => props.onZoom(1)}>100%</button>
      <button aria-label="Zoom in" onClick={() => props.onZoom(props.zoom + 0.1)}>+</button>
      <output>{Math.round(props.zoom * 100)}%</output>
    </div>
  </div>;
}
