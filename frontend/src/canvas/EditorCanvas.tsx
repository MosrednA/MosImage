import { CSSProperties, MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent, ReactNode, useEffect, useRef, useState } from "react";
import { FaceBox, SelectionPoint } from "../api";
import { Icon } from "../components/Icon";
import { SelectionActionProps, SelectionActions } from "../components/SelectionActions";
import { isTyping } from "../hooks/useWindowEvents";
import { OPACITY_RANGE, SOFTNESS_RANGE, TONE_RANGE } from "../layers/brush";
import { DEFAULT_EXPAND, autoExpand, autoSoften } from "../masks/maskRefine";
import { spotStyle } from "./barSpot";
import { ImageBox, paddedContextBox } from "./context";
import { ImageSize, viewportPointToImage } from "./coordinates";
import { BRUSH_RANGE, CanvasToolbar } from "./CanvasToolbar";
import { MaskBrushBar } from "./MaskBrushBar";
import { SelectionBar } from "./SelectionBar";
import { SKETCH_COLOURS, SKETCH_SIZE } from "./sketch";
import { CanvasTool, LAYER_STATUS, TOOLS, isMaskTool } from "./tools";
import { useBarSpot } from "./useBarSpot";
import { useMaskLayer } from "./useMaskLayer";
import { useSketchLayer } from "./useSketchLayer";
import { useViewport } from "./useViewport";

export type { CanvasTool } from "./tools";

/** What the layer stage needs from the editor view. */
export interface LayerView {
  zoom: number;
  size: ImageSize;
  brushSize: number;
  /** Mask brush hardness, opacity and tone (1 white, 0 black), 0–1. */
  brushHardness: number;
  brushOpacity: number;
  brushTone: number;
  hide: boolean;
  /** Pointer input goes to the layers instead of the AI selection. */
  interactive: boolean;
  trackCursor: (clientX: number, clientY: number) => void;
}

interface EditorCanvasProps {
  imageUrl: string | null;
  previewImageUrl?: string | null;
  reviewing: boolean;
  splitCompare?: boolean;
  imageName?: string;
  onMaskChange: (mask: Blob | null) => void;
  tool: CanvasTool;
  disabled: boolean;
  maskImportUrl: string | null;
  clearToken: number;
  points: SelectionPoint[];
  onObjectPoint: (point: SelectionPoint) => void;
  onRemovePoint: (index: number) => void;
  onClearSelection: () => void;
  onImageSize: (size: ImageSize) => void;
  onError: (message: string) => void;
  contextPadding: number;
  onContextPadding: (padding: number) => void;
  onHelp: () => void;
  /** A layer is selected: the tools edit its mask instead of the AI selection. */
  layerActive: boolean;
  layerPointCount: number;
  onUndoLayerPoint: () => void;
  renderLayers: (view: LayerView) => ReactNode;
  /** Next steps for a selection, shown beside it; invert, deselect and the grip come from the canvas. */
  selectionActions?: Omit<SelectionActionProps, "onInvert" | "onDeselect" | "grip"> | null;
  /** The sketch shows while Sketch is the operation; Apply clears it. */
  sketchVisible: boolean;
  sketchClearToken: number;
  onSketchChange: (sketch: Blob | null) => void;
  onFind: (text: string) => void;
  onSubject: () => void;
  /** The faces Fix faces found; clicking one picks it or leaves it out. */
  faces?: FaceOverlay | null;
}

export interface FaceOverlay {
  boxes: FaceBox[];
  picked: boolean[];
  finding: boolean;
  onToggle: (index: number) => void;
}

export function EditorCanvas(props: EditorCanvasProps) {
  const { imageUrl, previewImageUrl, reviewing, splitCompare, imageName, tool, disabled, points, contextPadding, faces } = props;
  const frameRef = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const surfaceRef = useRef<HTMLDivElement | null>(null);
  const cursorRef = useRef<HTMLDivElement | null>(null);
  const imageSize = useImageSize(imageUrl, props.onImageSize, props.onError);
  const [brushSize, setBrushSize] = useState(64);
  const [expandOverride, setExpandOverride] = useState<number | null>(null);
  const [softenOverride, setSoftenOverride] = useState<number | null>(null);
  const [showContext, setShowContext] = useState(true);
  const [maskVisible, setMaskVisible] = useState(true);
  const [subtract, setSubtract] = useState(false);
  const [spacePan, setSpacePan] = useState(false);
  const [layerHide, setLayerHide] = useState(true);
  const [softness, setSoftness] = useState<number>(SOFTNESS_RANGE.initial);
  const [tone, setTone] = useState<number>(TONE_RANGE.initial);
  const [brushOpacity, setBrushOpacity] = useState<number>(OPACITY_RANGE.initial);
  const [sketchColour, setSketchColour] = useState<string>(SKETCH_COLOURS[2]);
  const [sketchSize, setSketchSize] = useState<number>(SKETCH_SIZE.initial);
  const [sketchErase, setSketchErase] = useState(false);
  const [splitRatio, setSplitRatio] = useState(0.5);
  const { layerActive } = props;
  const expand = expandOverride ?? (imageSize ? autoExpand(imageSize.width, imageSize.height) : DEFAULT_EXPAND);
  const soften = softenOverride ?? (imageSize ? autoSoften(imageSize.width, imageSize.height) : 0);
  const panning = tool === "move" || spacePan;
  const viewport = useViewport(stageRef, surfaceRef, imageSize, panning);
  const { zoom, displaySize } = viewport;
  const mask = useMaskLayer({
    imageUrl, imageSize, disabled: disabled || layerActive, brushSize, edges: { expand, soften }, zoom,
    clearToken: props.clearToken, maskImportUrl: props.maskImportUrl, onMaskChange: props.onMaskChange, onError: props.onError,
  });
  const sketch = useSketchLayer({
    imageUrl, imageSize, colour: sketchColour, size: sketchSize, erase: sketchErase, clearToken: props.sketchClearToken, onChange: props.onSketchChange,
  });
  const contextBox = mask.selectionBounds && imageSize ? paddedContextBox(mask.selectionBounds, imageSize, contextPadding) : null;
  const editing = !reviewing && !!imageSize;
  const layerInput = (layerActive || tool === "transform") && !panning;
  const sketching = tool === "sketch" && !panning;
  const actionsSpot = useBarSpot(frameRef);

  useEffect(() => {
    const down = (event: KeyboardEvent) => {
      if (isTyping(event) || reviewing) return;
      const ctrl = event.ctrlKey || event.metaKey;
      if (event.code === "Space" && !ctrl) {
        event.preventDefault();
        if (!event.repeat) setSpacePan(true);
        return;
      }
      if (ctrl && event.key === "0") { event.preventDefault(); viewport.fit(); return; }
      if (ctrl && event.key === "1") { event.preventDefault(); viewport.setZoom(1); return; }
      if (disabled) return;
      if (ctrl && event.key.toLowerCase() === "i") { event.preventDefault(); mask.invert(); return; }
      if (ctrl || event.altKey) return;
      if (event.key === "[" || event.key === "]") {
        const factor = event.key === "]" ? 1.2 : 1 / 1.2;
        if (tool === "sketch") setSketchSize(size => Math.round(Math.min(Math.max(size * factor, SKETCH_SIZE.min), SKETCH_SIZE.max)));
        else setBrushSize(size => Math.round(Math.min(Math.max(size * factor, BRUSH_RANGE.min), BRUSH_RANGE.max)));
      } else if (event.key.toLowerCase() === "q") setMaskVisible(visible => !visible);
      // X swaps the mask brush between white and black, as in Photoshop.
      else if (layerActive && tool === "brush" && event.key.toLowerCase() === "x") setTone(value => TONE_RANGE.max - value);
      else if (layerActive) return;
      else if (event.key === "Escape" && (mask.hasSelection || points.length)) props.onClearSelection();
      else if (event.key === "Backspace" && tool === "object" && points.length) { event.preventDefault(); props.onRemovePoint(points.length - 1); }
    };
    const up = (event: KeyboardEvent) => { if (event.code === "Space") setSpacePan(false); };
    const blur = () => setSpacePan(false);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", blur);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); window.removeEventListener("blur", blur); };
  });

  function imagePoint(event: ReactMouseEvent<HTMLCanvasElement>): SelectionPoint | null {
    if (!imageSize) return null;
    const point = viewportPointToImage({ x: event.clientX, y: event.clientY }, event.currentTarget.getBoundingClientRect(), imageSize);
    return { x: Math.min(point.x, imageSize.width - 1), y: Math.min(point.y, imageSize.height - 1), label: 1 };
  }

  function beginGesture(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (disabled || panning || event.button !== 0 || !imageSize) return;
    if (tool === "object") {
      const point = imagePoint(event);
      if (point) props.onObjectPoint({ ...point, label: event.altKey ? 0 : 1 });
      return;
    }
    if (isMaskTool(tool)) mask.gesture.begin(event, tool, subtract);
  }

  function excludePoint(event: ReactMouseEvent<HTMLCanvasElement>) {
    if (tool !== "object") return;
    event.preventDefault();
    const point = imagePoint(event);
    if (point && !disabled) props.onObjectPoint({ ...point, label: 0 });
  }

  function trackCursor(clientX: number, clientY: number) {
    const cursor = cursorRef.current, surface = surfaceRef.current?.getBoundingClientRect();
    if (cursor && surface) cursor.style.transform = `translate(${clientX - surface.left}px, ${clientY - surface.top}px)`;
  }

  function moveCursor(event: ReactPointerEvent<HTMLCanvasElement>) {
    trackCursor(event.clientX, event.clientY);
    mask.gesture.update(event);
  }

  function updateSplit(clientX: number) {
    const rect = surfaceRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return;
    setSplitRatio(Math.min(1, Math.max(0, (clientX - rect.left) / rect.width)));
  }

  const brushCursor = editing && !panning && !disabled && (tool === "brush" || tool === "eraser" || tool === "sketch");
  const cursorSize = tool === "sketch" ? sketchSize : brushSize;
  const splitting = !!(reviewing && splitCompare && previewImageUrl);
  // Beside the selection, or where the user moved it in the frame around the image.
  const selectionActions = editing && !reviewing && !layerActive && !mask.drawing && maskVisible && mask.selectionBounds && props.selectionActions
    ? <SelectionActions {...props.selectionActions} onInvert={mask.invert} onDeselect={props.onClearSelection} grip={actionsSpot.grip} />
    : null;

  return (
    <section className="editor-canvas" aria-label="Image editor">
      <CanvasToolbar tool={tool} reviewing={reviewing} disabled={disabled}
        brushSize={brushSize} onBrushSize={setBrushSize}
        subtract={layerActive ? layerHide : subtract} onSubtract={layerActive ? setLayerHide : setSubtract} modeLabels={layerActive ? ["Restore", "Hide"] : undefined}
        pointCount={layerActive ? props.layerPointCount : points.length} onUndoPoint={layerActive ? props.onUndoLayerPoint : () => props.onRemovePoint(points.length - 1)}
        sketch={{ colour: sketchColour, onColour: setSketchColour, size: sketchSize, onSize: setSketchSize, erase: sketchErase, onErase: setSketchErase,
          canUndo: sketch.canUndo, onUndo: sketch.undo, canClear: sketch.hasStrokes, onClear: sketch.clear }}
        onFind={props.onFind} onSubject={props.onSubject}
        zoom={zoom} fitToStage={viewport.fitToStage} atActualSize={!viewport.fitToStage && viewport.manualZoom === 1} onZoom={value => viewport.setZoom(value)} onFit={viewport.fit} />
      {editing && layerActive && (tool === "brush" || tool === "eraser") && <MaskBrushBar disabled={disabled} erasing={tool === "eraser"}
        softness={softness} onSoftness={setSoftness} tone={tone} onTone={setTone} opacity={brushOpacity} onOpacity={setBrushOpacity} />}
      {editing && mask.hasSelection && <SelectionBar disabled={disabled}
        expand={expand} onExpand={setExpandOverride} soften={soften} softenAuto={softenOverride === null} onSoften={setSoftenOverride}
        contextPadding={contextPadding} onContextPadding={value => { setShowContext(true); props.onContextPadding(value); }}
        showContext={showContext} onToggleContext={() => setShowContext(value => !value)}
        maskVisible={maskVisible} onToggleMask={() => setMaskVisible(value => !value)}
        canUndo={mask.canUndo} canRedo={mask.canRedo} onUndo={mask.undo} onRedo={mask.redo} onInvert={mask.invert} onClear={props.onClearSelection} />}

      <div ref={frameRef} className="editor-stage-frame">
        <div ref={stageRef} className="editor-stage" data-tool={panning ? "move" : tool} {...viewport.panHandlers}>
          {imageUrl && imageSize && displaySize ? (
            <div ref={surfaceRef} className={"editor-surface" + (brushCursor ? " brush-cursor-active" : "")} style={{ width: displaySize.width, height: displaySize.height }}>
              <img src={splitting ? imageUrl : (previewImageUrl || imageUrl)} alt={imageName || "Source image"} draggable={false} />
              {props.renderLayers({ zoom, size: imageSize, brushSize, brushHardness: 1 - softness / 100, brushOpacity: brushOpacity / 100, brushTone: tone / 100, hide: layerHide, interactive: layerInput && editing, trackCursor })}
              {splitting && <>
                <img className="split-preview-image" src={previewImageUrl!} alt="Proposal preview" draggable={false}
                  style={{ clipPath: `inset(0 0 0 ${(splitRatio * 100).toFixed(2)}%)` }} />
                <div className="split-compare-overlay" aria-label="Split comparison"
                  onPointerDown={event => { event.stopPropagation(); event.currentTarget.setPointerCapture(event.pointerId); updateSplit(event.clientX); }}
                  onPointerMove={event => { if (event.buttons & 1) updateSplit(event.clientX); }}>
                  <div className="split-compare-divider" style={{ left: `${(splitRatio * 100).toFixed(2)}%` }}>
                    <span className="split-compare-label before">Original</span>
                    <span className="split-compare-handle" aria-hidden="true">↔</span>
                    <span className="split-compare-label after">Proposal</span>
                  </div>
                </div>
              </>}
              <canvas ref={sketch.canvasRef} className="sketch-draw" width={imageSize.width} height={imageSize.height} data-tool="sketch" aria-label="Sketch canvas"
                style={{ pointerEvents: sketching && editing && !disabled ? undefined : "none", opacity: props.sketchVisible && !reviewing ? 1 : 0 }}
                onPointerDown={event => { if (event.button === 0 && !disabled) sketch.gesture.begin(event); }}
                onPointerMove={event => { trackCursor(event.clientX, event.clientY); sketch.gesture.update(event); }}
                onPointerUp={sketch.gesture.finish} onPointerCancel={sketch.gesture.cancel} />
              <canvas
                ref={mask.canvasRef}
                className="mask-draw"
                width={imageSize.width}
                height={imageSize.height}
                data-tool={tool}
                aria-label="Selection canvas"
                style={{ pointerEvents: disabled || panning || layerInput || sketching ? "none" : undefined, opacity: mask.drawing && !reviewing ? undefined : 0 }}
                onPointerDown={beginGesture}
                onPointerMove={moveCursor}
                onPointerUp={mask.gesture.finish}
                onPointerCancel={mask.gesture.cancel}
                onContextMenu={excludePoint}
              />
              <canvas ref={mask.overlayRef} className="mask-overlay" width={imageSize.width} height={imageSize.height} aria-hidden="true"
                style={{ opacity: reviewing || mask.drawing || !maskVisible ? 0 : 1 }} />
              {selectionActions && !actionsSpot.spot && mask.selectionBounds
                && <div className="selection-actions-anchor" style={actionsPlacement(mask.selectionBounds, zoom, displaySize)}>{selectionActions}</div>}
              {brushCursor && <div ref={cursorRef} className="brush-cursor" style={{ width: cursorSize * zoom, height: cursorSize * zoom, margin: -cursorSize * zoom / 2 }} />}
              {!reviewing && showContext && contextBox && <div className="context-boundary" aria-label="Protected context boundary" style={{ left: contextBox.x * zoom, top: contextBox.y * zoom, width: contextBox.width * zoom, height: contextBox.height * zoom }}><span>Context · stays unchanged</span></div>}
              {!reviewing && maskVisible && !layerActive && points.map((point, index) => <button key={`${index}-${point.x}-${point.y}`} type="button"
                className={`selection-point ${point.label ? "positive" : "negative"}`} style={{ left: point.x * zoom, top: point.y * zoom }}
                disabled={disabled} aria-label={`Remove ${point.label ? "include" : "exclude"} point ${index + 1}`} title="Click to remove this point"
                onPointerDown={event => event.stopPropagation()} onClick={() => props.onRemovePoint(index)}>
                <span aria-hidden="true">{point.label ? "+" : "−"}</span>
              </button>)}
              {!reviewing && faces && faces.boxes.map(([left, top, right, bottom], index) => <button key={`${index}-${left}-${top}`} type="button"
                className={"face-box" + (faces.picked[index] ? " picked" : "")} style={{ left: left * zoom, top: top * zoom, width: (right - left) * zoom, height: (bottom - top) * zoom }}
                disabled={disabled} aria-pressed={faces.picked[index]} aria-label={`Face ${index + 1}`} title={faces.picked[index] ? "Click to leave this face out" : "Click to fix this face too"}
                onPointerDown={event => event.stopPropagation()} onClick={() => faces.onToggle(index)}>
                <span aria-hidden="true">{faces.picked[index] ? "✓" : "+"}</span>
              </button>)}
              {!reviewing && faces?.finding && <div className="face-status" role="status">Finding faces…</div>}
            </div>
          ) : (
            <div className="canvas-empty">
              <span className="empty-mark">+</span>
              <strong>Open, drop, or choose a recent image</strong>
              <span>Your mask stays aligned to the image at its native resolution.</span>
            </div>
          )}
        </div>
        {selectionActions && actionsSpot.spot && <div className="selection-actions-anchor" style={spotStyle(actionsSpot.spot)}>{selectionActions}</div>}
      </div>

      <footer className="canvas-status">
        <span>{imageSize ? `${imageSize.width} × ${imageSize.height} px` : "No document"}</span>
        <span>{reviewing ? "← → to compare proposals · Apply to save" : spacePan ? "Drag to pan" : (layerActive ? LAYER_STATUS[tool] : TOOLS[tool].status)}</span>
        <button className="status-help" onClick={props.onHelp} title="Keyboard shortcuts · ?"><Icon name="help" /><span>Shortcuts</span></button>
      </footer>
    </section>
  );
}

/** Just below the selection, or above it when there is no room; inside it when it fills the picture. */
function actionsPlacement(bounds: ImageBox, zoom: number, display: { width: number; height: number }): CSSProperties {
  const room = 52;
  const left = Math.min(Math.max((bounds.x + bounds.width / 2) * zoom, 0), display.width);
  const below = (bounds.y + bounds.height) * zoom + 10, above = bounds.y * zoom - 10;
  if (below + room <= display.height) return { left, top: below };
  if (above >= room) return { left, top: above, transform: "translate(-50%, -100%)" };
  return { left, top: display.height - 10, transform: "translate(-50%, -100%)" };
}

function useImageSize(imageUrl: string | null, onImageSize: (size: ImageSize) => void, onError: (message: string) => void): ImageSize | null {
  const [imageSize, setImageSize] = useState<ImageSize | null>(null);
  useEffect(() => {
    let alive = true;
    setImageSize(null);
    if (!imageUrl) return;
    const image = new Image();
    image.onload = () => {
      if (!alive) return;
      const size = { width: image.naturalWidth, height: image.naturalHeight };
      setImageSize(size);
      onImageSize(size);
    };
    image.onerror = () => { if (alive) onError("Could not open this image."); };
    image.src = imageUrl;
    return () => { alive = false; };
  }, [imageUrl, onImageSize]);
  return imageSize;
}
