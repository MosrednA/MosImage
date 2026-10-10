import { FormEvent, useState } from "react";
import { Icon } from "../components/Icon";
import { SKETCH_COLOURS, SKETCH_SIZE } from "./sketch";
import { CanvasTool, TOOLS } from "./tools";
import { MOD } from "../platform";

/** Pencil options: colour, size, drawing or erasing, and the sketch's own undo. */
export interface SketchControls {
  colour: string;
  onColour: (colour: string) => void;
  size: number;
  onSize: (size: number) => void;
  erase: boolean;
  onErase: (erase: boolean) => void;
  canUndo: boolean;
  onUndo: () => void;
  canClear: boolean;
  onClear: () => void;
}

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
  sketch: SketchControls;
  /** Select by description, such as "the hat", and the main subject. */
  onFind: (text: string) => void;
  onSubject: () => void;
  zoom: number;
  fitToStage: boolean;
  atActualSize: boolean;
  onZoom: (zoom: number) => void;
  onFit: () => void;
}

export const BRUSH_RANGE = { min: 4, max: 400 } as const;

/** Options for the active tool, plus zoom. Selection edges live in SelectionBar. */
export function CanvasToolbar(props: Props) {
  const { tool, reviewing, disabled, sketch } = props;
  const [query, setQuery] = useState("");
  function find(event: FormEvent) {
    event.preventDefault();
    if (query.trim() && !disabled) props.onFind(query.trim());
  }
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
    {!reviewing && tool === "object" && <>
      <form className="find-object" onSubmit={find} title="Select everything that matches, in English">
        <Icon name="search" />
        <input aria-label="Find an object by description" placeholder="Find… e.g. the hat" value={query} maxLength={120} disabled={disabled} onChange={event => setQuery(event.target.value)} />
      </form>
      <button disabled={disabled} title="Select the main subject, with soft edges for hair and fur" onClick={props.onSubject}>Subject</button>
    </>}
    {!reviewing && tool === "object" && props.pointCount > 0 && <>
      <span className="point-count">{props.pointCount} {props.pointCount === 1 ? "point" : "points"}</span>
      <button disabled={disabled} title="Remove the last point · Backspace" onClick={props.onUndoPoint}>Undo point</button>
    </>}
    {!reviewing && tool === "sketch" && <>
      <div className="sketch-colours" role="radiogroup" aria-label="Sketch colour">
        {SKETCH_COLOURS.map(colour => <button key={colour} role="radio" aria-checked={sketch.colour === colour} aria-label={`Colour ${colour}`} disabled={disabled}
          style={{ background: colour }} onClick={() => { sketch.onColour(colour); sketch.onErase(false); }} />)}
        <label className="sketch-custom" title="Any colour" style={{ background: sketch.colour }}>
          <input type="color" aria-label="Pick any colour" value={sketch.colour} disabled={disabled} onChange={event => { sketch.onColour(event.target.value); sketch.onErase(false); }} />
        </label>
      </div>
      <label className="brush-control" title="Pencil size · [ and ]">
        <span>Size</span>
        <input type="range" min={SKETCH_SIZE.min} max={SKETCH_SIZE.max} value={sketch.size} disabled={disabled} onChange={event => sketch.onSize(Number(event.target.value))} />
        <output>{sketch.size}px</output>
      </label>
      <div className="segmented compact" aria-label="Pencil mode">
        <button aria-pressed={!sketch.erase} disabled={disabled} onClick={() => sketch.onErase(false)}>Draw</button>
        <button aria-pressed={sketch.erase} disabled={disabled} title="Or hold Alt" onClick={() => sketch.onErase(true)}>Erase</button>
      </div>
      <button disabled={disabled || !sketch.canUndo} title="Undo the last stroke" onClick={sketch.onUndo}>Undo stroke</button>
      <button disabled={disabled || !sketch.canClear} title="Remove the whole sketch" onClick={sketch.onClear}>Clear</button>
    </>}
    <div className="zoom-controls">
      <button aria-label="Zoom out" title={`Zoom out · ${MOD} + scroll`} onClick={() => props.onZoom(props.zoom / 1.25)}>−</button>
      <button className={props.fitToStage ? "selected" : ""} title={`Fit to window · ${MOD} + 0`} onClick={props.onFit}>Fit</button>
      <button className={props.atActualSize ? "selected" : ""} title={`Actual pixels · ${MOD} + 1`} onClick={() => props.onZoom(1)}>100%</button>
      <button aria-label="Zoom in" title={`Zoom in · ${MOD} + scroll`} onClick={() => props.onZoom(props.zoom * 1.25)}>+</button>
      <output>{Math.round(props.zoom * 100)}%</output>
    </div>
  </div>;
}
