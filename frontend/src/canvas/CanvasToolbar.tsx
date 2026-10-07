import { CanvasTool, TOOLS } from "./tools";

interface Props {
  tool: CanvasTool;
  reviewing: boolean;
  disabled: boolean;
  brushSize: number;
  onBrushSize: (size: number) => void;
  subtract: boolean;
  onSubtract: (subtract: boolean) => void;
  /** Names for the two Lasso/Box modes; the second is the subtracting one. */
  modeLabels?: [string, string];
  pointCount: number;
  onUndoPoint: () => void;
  zoom: number;
  fitToStage: boolean;
  atActualSize: boolean;
  onZoom: (zoom: number) => void;
  onFit: () => void;
}

export const BRUSH_RANGE = { min: 4, max: 400 } as const;

/** Options for the active tool, plus zoom. Selection edges live in SelectionBar. */
export function CanvasToolbar(props: Props) {
  const { tool, reviewing, disabled } = props;
  return <div className="editor-toolbar">
    <span className="canvas-context-label">{reviewing ? "Review your proposal" : TOOLS[tool].label}</span>
    {!reviewing && (tool === "brush" || tool === "eraser") && <label className="brush-control" title="Brush size · [ and ]">
      <span>Size</span>
      <input type="range" min={BRUSH_RANGE.min} max={BRUSH_RANGE.max} value={props.brushSize} disabled={disabled} onChange={event => props.onBrushSize(Number(event.target.value))} />
      <output>{props.brushSize}px</output>
    </label>}
    {!reviewing && (tool === "rectangle" || tool === "lasso") && <div className="segmented compact" aria-label="Selection mode">
      <button aria-pressed={!props.subtract} disabled={disabled} onClick={() => props.onSubtract(false)}>{props.modeLabels?.[0] ?? "Add"}</button>
      <button aria-pressed={props.subtract} disabled={disabled} title="Or hold Alt" onClick={() => props.onSubtract(true)}>{props.modeLabels?.[1] ?? "Subtract"}</button>
    </div>}
    {!reviewing && tool === "object" && props.pointCount > 0 && <>
      <span className="point-count">{props.pointCount} {props.pointCount === 1 ? "point" : "points"}</span>
      <button disabled={disabled} title="Remove the last point · Backspace" onClick={props.onUndoPoint}>Undo point</button>
    </>}
    <div className="zoom-controls">
      <button aria-label="Zoom out" title="Zoom out · Ctrl + scroll" onClick={() => props.onZoom(props.zoom / 1.25)}>−</button>
      <button className={props.fitToStage ? "selected" : ""} title="Fit to window · Ctrl + 0" onClick={props.onFit}>Fit</button>
      <button className={props.atActualSize ? "selected" : ""} title="Actual pixels · Ctrl + 1" onClick={() => props.onZoom(1)}>100%</button>
      <button aria-label="Zoom in" title="Zoom in · Ctrl + scroll" onClick={() => props.onZoom(props.zoom * 1.25)}>+</button>
      <output>{Math.round(props.zoom * 100)}%</output>
    </div>
  </div>;
}
