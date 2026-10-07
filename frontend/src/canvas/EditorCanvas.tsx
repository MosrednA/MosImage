import { MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent, useEffect, useRef, useState } from "react";
import { SelectionPoint } from "../api";
import { Icon } from "../components/Icon";
import { DEFAULT_EXPAND, autoSoften } from "../masks/maskRefine";
import { paddedContextBox } from "./context";
import { ImageSize, viewportPointToImage } from "./coordinates";
import { BRUSH_RANGE, CanvasToolbar } from "./CanvasToolbar";
import { SelectionBar } from "./SelectionBar";
import { CanvasTool, TOOLS, isMaskTool } from "./tools";
import { useMaskLayer } from "./useMaskLayer";
import { useViewport } from "./useViewport";

export type { CanvasTool } from "./tools";

interface EditorCanvasProps {
  imageUrl: string | null;
  previewImageUrl?: string | null;
  reviewing: boolean;
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
}

export function EditorCanvas(props: EditorCanvasProps) {
  const { imageUrl, previewImageUrl, reviewing, imageName, tool, disabled, points, contextPadding } = props;
  const stageRef = useRef<HTMLDivElement | null>(null);
  const surfaceRef = useRef<HTMLDivElement | null>(null);
  const cursorRef = useRef<HTMLDivElement | null>(null);
  const imageSize = useImageSize(imageUrl, props.onImageSize, props.onError);
  const [brushSize, setBrushSize] = useState(64);
  const [expand, setExpand] = useState(DEFAULT_EXPAND);
  const [softenOverride, setSoftenOverride] = useState<number | null>(null);
  const [showContext, setShowContext] = useState(true);
  const [maskVisible, setMaskVisible] = useState(true);
  const [subtract, setSubtract] = useState(false);
  const [spacePan, setSpacePan] = useState(false);
  const soften = softenOverride ?? (imageSize ? autoSoften(imageSize.width, imageSize.height) : 0);
  const panning = tool === "move" || spacePan;
  const viewport = useViewport(stageRef, surfaceRef, imageSize, panning);
  const { zoom, displaySize } = viewport;
  const mask = useMaskLayer({
    imageUrl, imageSize, disabled, brushSize, edges: { expand, soften }, zoom,
    clearToken: props.clearToken, maskImportUrl: props.maskImportUrl, onMaskChange: props.onMaskChange, onError: props.onError,
  });
  const contextBox = mask.selectionBounds && imageSize ? paddedContextBox(mask.selectionBounds, imageSize, contextPadding) : null;
  const editing = !reviewing && !!imageSize;

  useEffect(() => {
    const typing = (target: EventTarget | null) => target instanceof HTMLElement && !!target.closest("input, textarea, select, [contenteditable], dialog");
    const down = (event: KeyboardEvent) => {
      if (typing(event.target) || reviewing) return;
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
        setBrushSize(size => Math.round(Math.min(Math.max(size * factor, BRUSH_RANGE.min), BRUSH_RANGE.max)));
      } else if (event.key.toLowerCase() === "q") setMaskVisible(visible => !visible);
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

  function moveCursor(event: ReactPointerEvent<HTMLCanvasElement>) {
    const cursor = cursorRef.current, surface = surfaceRef.current?.getBoundingClientRect();
    if (cursor && surface) cursor.style.transform = `translate(${event.clientX - surface.left}px, ${event.clientY - surface.top}px)`;
    mask.gesture.update(event);
  }

  const brushCursor = editing && !panning && !disabled && (tool === "brush" || tool === "eraser");

  return (
    <section className="editor-canvas" aria-label="Image editor">
      <CanvasToolbar tool={tool} reviewing={reviewing} disabled={disabled}
        brushSize={brushSize} onBrushSize={setBrushSize} subtract={subtract} onSubtract={setSubtract}
        pointCount={points.length} onUndoPoint={() => props.onRemovePoint(points.length - 1)}
        zoom={zoom} fitToStage={viewport.fitToStage} atActualSize={!viewport.fitToStage && viewport.manualZoom === 1} onZoom={value => viewport.setZoom(value)} onFit={viewport.fit} />
      {editing && mask.hasSelection && <SelectionBar disabled={disabled}
        expand={expand} onExpand={setExpand} soften={soften} softenAuto={softenOverride === null} onSoften={setSoftenOverride}
        contextPadding={contextPadding} onContextPadding={value => { setShowContext(true); props.onContextPadding(value); }}
        showContext={showContext} onToggleContext={() => setShowContext(value => !value)}
        maskVisible={maskVisible} onToggleMask={() => setMaskVisible(value => !value)}
        canUndo={mask.canUndo} canRedo={mask.canRedo} onUndo={mask.undo} onRedo={mask.redo} onInvert={mask.invert} onClear={props.onClearSelection} />}

      <div ref={stageRef} className="editor-stage" data-tool={panning ? "move" : tool} {...viewport.panHandlers}>
        {imageUrl && imageSize && displaySize ? (
          <div ref={surfaceRef} className={"editor-surface" + (brushCursor ? " brush-cursor-active" : "")} style={{ width: displaySize.width, height: displaySize.height }}>
            <img src={previewImageUrl || imageUrl} alt={imageName || "Source image"} draggable={false} />
            <canvas
              ref={mask.canvasRef}
              className="mask-draw"
              width={imageSize.width}
              height={imageSize.height}
              data-tool={tool}
              aria-label="Selection canvas"
              style={{ pointerEvents: disabled || panning ? "none" : undefined, opacity: mask.drawing && !reviewing ? undefined : 0 }}
              onPointerDown={beginGesture}
              onPointerMove={moveCursor}
              onPointerUp={mask.gesture.finish}
              onPointerCancel={mask.gesture.cancel}
              onContextMenu={excludePoint}
            />
            <canvas ref={mask.overlayRef} className="mask-overlay" width={imageSize.width} height={imageSize.height} aria-hidden="true"
              style={{ opacity: reviewing || mask.drawing || !maskVisible ? 0 : 1 }} />
            {brushCursor && <div ref={cursorRef} className="brush-cursor" style={{ width: brushSize * zoom, height: brushSize * zoom, margin: -brushSize * zoom / 2 }} />}
            {!reviewing && showContext && contextBox && <div className="context-boundary" aria-label="Protected context boundary" style={{ left: contextBox.x * zoom, top: contextBox.y * zoom, width: contextBox.width * zoom, height: contextBox.height * zoom }}><span>Context · stays unchanged</span></div>}
            {!reviewing && maskVisible && points.map((point, index) => <button key={`${index}-${point.x}-${point.y}`} type="button"
              className={`selection-point ${point.label ? "positive" : "negative"}`} style={{ left: point.x * zoom, top: point.y * zoom }}
              disabled={disabled} aria-label={`Remove ${point.label ? "include" : "exclude"} point ${index + 1}`} title="Click to remove this point"
              onPointerDown={event => event.stopPropagation()} onClick={() => props.onRemovePoint(index)}>
              <span aria-hidden="true">{point.label ? "+" : "−"}</span>
            </button>)}
          </div>
        ) : (
          <div className="canvas-empty">
            <span className="empty-mark">+</span>
            <strong>Open, drop, or choose a recent image</strong>
            <span>Your mask stays aligned to the image at its native resolution.</span>
          </div>
        )}
      </div>

      <footer className="canvas-status">
        <span>{imageSize ? `${imageSize.width} × ${imageSize.height} px` : "No document"}</span>
        <span>{reviewing ? "← → to compare proposals · Apply to save" : spacePan ? "Drag to pan" : TOOLS[tool].status}</span>
        <button className="status-help" onClick={props.onHelp} title="Keyboard shortcuts · ?"><Icon name="help" /><span>Shortcuts</span></button>
      </footer>
    </section>
  );
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
