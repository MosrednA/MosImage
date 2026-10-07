import { PointerEvent as ReactPointerEvent, useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  featherMaskPixels,
  growMaskPixels,
  invertMaskPixels,
  shrinkMaskPixels,
} from "../masks/maskOperations";
import { ImageSize, Point, viewportPointToImage } from "./coordinates";
import { SelectionPoint } from "../api";
import { ImageBox, maskBounds, paddedContextBox } from "./context";

type MaskTool = "brush" | "eraser" | "rectangle" | "lasso";
export type CanvasTool = MaskTool | "move" | "object";
type SelectionOperation = "add" | "subtract";

interface EditorCanvasProps {
  imageUrl: string | null;
  previewImageUrl?: string | null;
  reviewing: boolean;
  imageName?: string;
  onMaskChange: (mask: Blob | null) => void;
  tool: CanvasTool;
  onToolChange: (tool: CanvasTool) => void;
  disabled: boolean;
  maskImportUrl: string | null;
  clearToken: number;
  points: SelectionPoint[];
  onObjectPoint: (point: SelectionPoint) => void;
  onImageSize: (size: ImageSize) => void;
  onError: (message: string) => void;
  contextPadding: number;
}

interface MaskSnapshot {
  bytes: number;
  imageData: ImageData;
}

interface PointerGesture {
  pointerId: number;
  tool: MaskTool;
  operation: SelectionOperation;
  start: Point;
  points: Point[];
  base: ImageData;
}

interface PanGesture {
  pointerId: number;
  clientX: number;
  clientY: number;
  scrollLeft: number;
  scrollTop: number;
}

const MAX_HISTORY_BYTES = 96 * 1024 * 1024;

export function EditorCanvas({ imageUrl, previewImageUrl, reviewing, imageName, onMaskChange, tool, onToolChange, disabled, maskImportUrl, clearToken, points, onObjectPoint, onImageSize, onError, contextPadding }: EditorCanvasProps) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const gestureRef = useRef<PointerGesture | null>(null);
  const panRef = useRef<PanGesture | null>(null);
  const undoStackRef = useRef<MaskSnapshot[]>([]);
  const redoStackRef = useRef<MaskSnapshot[]>([]);
  const [imageSize, setImageSize] = useState<ImageSize | null>(null);
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 });
  const maskVersionRef = useRef(0);
  const [brushSize, setBrushSize] = useState(64);
  const [edgeSize, setEdgeSize] = useState(8);
  const [hasSelection, setHasSelection] = useState(false);
  const [selectionBounds, setSelectionBounds] = useState<ImageBox | null>(null);
  const [showContext, setShowContext] = useState(true);
  const [fitToStage, setFitToStage] = useState(true);
  const [manualZoom, setManualZoom] = useState(1);
  const [historyVersion, setHistoryVersion] = useState(0);

  useLayoutEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;

    const updateSize = () => setStageSize({ width: stage.clientWidth, height: stage.clientHeight });
    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(stage);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let alive = true;
    maskVersionRef.current += 1;
    gestureRef.current = null;
    undoStackRef.current = [];
    redoStackRef.current = [];
    setHistoryVersion((version) => version + 1);
    setHasSelection(false);
    setSelectionBounds(null);
    setImageSize(null);
    onMaskChange(null);
    if (!imageUrl) return;

    const image = new Image();
    image.onload = () => {
      if (!alive) return;
      const size = { width: image.naturalWidth, height: image.naturalHeight };
      setImageSize(size);
      onImageSize(size);
      setFitToStage(true);
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = size.width;
      canvas.height = size.height;
      canvas.getContext("2d")?.clearRect(0, 0, size.width, size.height);
    };
    image.onerror = () => { if (alive) onError("Could not open this image."); };
    image.src = imageUrl;
    return () => { alive = false; };
  }, [imageUrl, onMaskChange, onImageSize]);

  useEffect(() => {
    if (!clearToken) return;
    clearMask();
  }, [clearToken]);

  useEffect(() => {
    const handleUndo = (event: KeyboardEvent) => {
      if (disabled || !(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== "z") return;
      if (event.target instanceof HTMLElement && event.target.closest("input, textarea, select, dialog")) return;
      const stack = event.shiftKey ? redoStackRef.current : undoStackRef.current;
      if (!stack.length) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      if (event.shiftKey) redo(); else undo();
    };
    window.addEventListener("keydown", handleUndo, true);
    return () => window.removeEventListener("keydown", handleUndo, true);
  }, [disabled, imageUrl]);

  useEffect(() => {
    if (!maskImportUrl || !imageSize) return;
    let alive = true;
    let url: string | null = null;
    void fetch(maskImportUrl).then((response) => {
      if (!response.ok) throw new Error("Could not load selection");
      return response.blob();
    }).then((blob) => {
      if (!alive) return;
      url = URL.createObjectURL(blob);
      const image = new Image();
      image.onload = () => {
        const canvas = canvasRef.current;
        const context = canvas?.getContext("2d", { willReadFrequently: true });
        if (!alive) return;
        if (!canvas || !context || image.width !== canvas.width || image.height !== canvas.height) {
          onError("The selection does not match the current image. Select the object again.");
          return;
        }
        rememberMask();
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.drawImage(image, 0, 0);
        const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
        for (let i = 0; i < pixels.data.length; i += 4) {
          pixels.data[i + 3] = pixels.data[i];
          pixels.data[i] = pixels.data[i + 1] = pixels.data[i + 2] = 255;
        }
        context.putImageData(pixels, 0, 0);
        emitMask();
      };
      image.onerror = () => { if (alive) onError("Could not load the selection. Select the object again."); };
      image.src = url;
    }).catch(() => { if (alive) { onMaskChange(null); onError("Could not load the selection. Select the object again."); } });
    return () => { alive = false; if (url) URL.revokeObjectURL(url); };
  }, [maskImportUrl, imageSize]);

  const fitZoom = imageSize
    ? Math.min(
        Math.max((stageSize.width - 40) / imageSize.width, 0.05),
        Math.max((stageSize.height - 40) / imageSize.height, 0.05),
        1,
      )
    : 1;
  const zoom = fitToStage ? fitZoom : manualZoom;
  const contextBox = selectionBounds && imageSize ? paddedContextBox(selectionBounds, imageSize, contextPadding) : null;
  const displaySize = imageSize
    ? { width: Math.round(imageSize.width * zoom), height: Math.round(imageSize.height * zoom) }
    : null;

  function captureMask(): ImageData | null {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d", { willReadFrequently: true });
    if (!canvas || !context) return null;
    return context.getImageData(0, 0, canvas.width, canvas.height);
  }

  function rememberMask(): ImageData | null {
    const imageData = captureMask();
    if (!imageData) return null;
    undoStackRef.current.push({ imageData, bytes: imageData.data.byteLength });
    trimHistory(undoStackRef.current);
    redoStackRef.current = [];
    setHistoryVersion((version) => version + 1);
    return imageData;
  }

  function pointFor(event: ReactPointerEvent<HTMLCanvasElement>): Point | null {
    const canvas = canvasRef.current;
    if (!canvas || !imageSize) return null;
    return viewportPointToImage(
      { x: event.clientX, y: event.clientY },
      canvas.getBoundingClientRect(),
      imageSize,
    );
  }

  function beginGesture(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (disabled || tool === "move" || event.button !== 0) return;
    const point = pointFor(event);
    if (tool === "object") {
      if (point && imageSize) onObjectPoint({ x: Math.min(point.x, imageSize.width - 1), y: Math.min(point.y, imageSize.height - 1), label: event.altKey ? 0 : 1 });
      return;
    }
    const base = rememberMask();
    if (!point || !base) return;

    const gesture: PointerGesture = {
      pointerId: event.pointerId,
      tool,
      operation: tool === "eraser" || event.altKey ? "subtract" : "add",
      start: point,
      points: [point],
      base,
    };
    gestureRef.current = gesture;
    event.currentTarget.setPointerCapture(event.pointerId);

    if (tool === "brush" || tool === "eraser") drawBrushSegment(gesture, point);
  }

  function updateGesture(event: ReactPointerEvent<HTMLCanvasElement>) {
    const gesture = gestureRef.current;
    const point = pointFor(event);
    if (!gesture || gesture.pointerId !== event.pointerId || !point) return;

    if (gesture.tool === "brush" || gesture.tool === "eraser") {
      drawBrushSegment(gesture, point);
      return;
    }

    if (gesture.tool === "rectangle") {
      gesture.points = [gesture.start, point];
    } else {
      const previous = gesture.points[gesture.points.length - 1];
      if (Math.hypot(point.x - previous.x, point.y - previous.y) >= 1) gesture.points.push(point);
    }
    renderSelectionGesture(gesture);
  }

  function finishGesture(event: ReactPointerEvent<HTMLCanvasElement>) {
    const gesture = gestureRef.current;
    if (!gesture || gesture.pointerId !== event.pointerId) return;
    updateGesture(event);

    const meaningful = isMeaningfulGesture(gesture);
    if (!meaningful) {
      canvasRef.current?.getContext("2d")?.putImageData(gesture.base, 0, 0);
      undoStackRef.current.pop();
      setHistoryVersion((version) => version + 1);
    }
    gestureRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    emitMask();
  }

  function cancelGesture(event: ReactPointerEvent<HTMLCanvasElement>) {
    const gesture = gestureRef.current;
    if (!gesture || gesture.pointerId !== event.pointerId) return;
    canvasRef.current?.getContext("2d")?.putImageData(gesture.base, 0, 0);
    undoStackRef.current.pop();
    gestureRef.current = null;
    setHistoryVersion((version) => version + 1);
    emitMask();
  }

  function drawBrushSegment(gesture: PointerGesture, point: Point) {
    const context = canvasRef.current?.getContext("2d");
    if (!context) return;
    const previous = gesture.points[gesture.points.length - 1] ?? point;
    context.save();
    context.globalCompositeOperation = gesture.operation === "subtract" ? "destination-out" : "source-over";
    context.strokeStyle = "white";
    context.fillStyle = "white";
    context.lineCap = "round";
    context.lineJoin = "round";
    context.lineWidth = brushSize;
    context.beginPath();
    context.moveTo(previous.x, previous.y);
    context.lineTo(point.x, point.y);
    context.stroke();
    context.beginPath();
    context.arc(point.x, point.y, brushSize / 2, 0, Math.PI * 2);
    context.fill();
    context.restore();
    gesture.points.push(point);
  }

  function renderSelectionGesture(gesture: PointerGesture) {
    const context = canvasRef.current?.getContext("2d");
    if (!context) return;
    context.putImageData(gesture.base, 0, 0);
    context.save();
    context.globalCompositeOperation = gesture.operation === "subtract" ? "destination-out" : "source-over";
    context.fillStyle = "white";
    context.beginPath();

    if (gesture.tool === "rectangle") {
      const end = gesture.points[1] ?? gesture.start;
      context.rect(gesture.start.x, gesture.start.y, end.x - gesture.start.x, end.y - gesture.start.y);
    } else {
      context.moveTo(gesture.points[0].x, gesture.points[0].y);
      for (const point of gesture.points.slice(1)) context.lineTo(point.x, point.y);
      context.closePath();
    }
    context.fill();
    context.restore();
  }

  function applyMaskTransform(transform: "invert" | "grow" | "shrink" | "feather") {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d", { willReadFrequently: true });
    const source = rememberMask();
    if (!canvas || !context || !source) return;

    const pixels = { width: canvas.width, height: canvas.height, data: source.data };
    const transformed =
      transform === "invert"
        ? invertMaskPixels(pixels)
        : transform === "grow"
          ? growMaskPixels(pixels, edgeSize)
          : transform === "shrink"
            ? shrinkMaskPixels(pixels, edgeSize)
            : featherMaskPixels(pixels, edgeSize);
    const result = context.createImageData(canvas.width, canvas.height);
    result.data.set(transformed);
    context.putImageData(result, 0, 0);
    emitMask();
  }

  function clearMask() {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context || !hasSelection) return;
    rememberMask();
    context.clearRect(0, 0, canvas.width, canvas.height);
    emitMask();
  }

  function undo() {
    const snapshot = undoStackRef.current.pop();
    if (!snapshot) return;
    pushCurrentMask(redoStackRef.current);
    restoreSnapshot(snapshot);
  }

  function redo() {
    const snapshot = redoStackRef.current.pop();
    if (!snapshot) return;
    pushCurrentMask(undoStackRef.current);
    restoreSnapshot(snapshot);
  }

  function pushCurrentMask(stack: MaskSnapshot[]) {
    const imageData = captureMask();
    if (!imageData) return;
    stack.push({ imageData, bytes: imageData.data.byteLength });
    trimHistory(stack);
  }

  function restoreSnapshot(snapshot: MaskSnapshot) {
    const context = canvasRef.current?.getContext("2d");
    if (!context) return;
    context.putImageData(snapshot.imageData, 0, 0);
    setHistoryVersion((version) => version + 1);
    emitMask();
  }

  function emitMask() {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d", { willReadFrequently: true });
    if (!canvas || !context) return;
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
    const bounds = maskBounds(pixels);
    setSelectionBounds(bounds);
    const selected = bounds !== null;
    setHasSelection(selected);
    if (!selected) {
      maskVersionRef.current += 1;
      onMaskChange(null);
      return;
    }
    const version = ++maskVersionRef.current;
    canvas.toBlob((blob) => { if (version === maskVersionRef.current) onMaskChange(blob); }, "image/png");
  }

  function changeZoom(nextZoom: number) {
    setFitToStage(false);
    setManualZoom(Math.min(Math.max(nextZoom, 0.1), 2));
  }

  function beginPan(event: ReactPointerEvent<HTMLDivElement>) {
    if (tool !== "move" && event.button !== 1) return;
    const stage = stageRef.current;
    if (!stage) return;
    event.preventDefault();
    panRef.current = {
      pointerId: event.pointerId,
      clientX: event.clientX,
      clientY: event.clientY,
      scrollLeft: stage.scrollLeft,
      scrollTop: stage.scrollTop,
    };
    stage.setPointerCapture(event.pointerId);
  }

  function updatePan(event: ReactPointerEvent<HTMLDivElement>) {
    const pan = panRef.current;
    const stage = stageRef.current;
    if (!pan || !stage || pan.pointerId !== event.pointerId) return;
    stage.scrollLeft = pan.scrollLeft - (event.clientX - pan.clientX);
    stage.scrollTop = pan.scrollTop - (event.clientY - pan.clientY);
  }

  function finishPan(event: ReactPointerEvent<HTMLDivElement>) {
    const pan = panRef.current;
    const stage = stageRef.current;
    if (!pan || !stage || pan.pointerId !== event.pointerId) return;
    panRef.current = null;
    if (stage.hasPointerCapture(event.pointerId)) stage.releasePointerCapture(event.pointerId);
  }

  return (
    <section className="editor-canvas" aria-label="Image editor">
      <div className="editor-toolbar">
        {!reviewing && hasSelection && <button aria-pressed={showContext} className={showContext ? "selected" : ""} title="Show the protected context sent with your selection" onClick={() => setShowContext(value => !value)}>Context</button>}
        {reviewing ? <span className="canvas-context-label">Review your proposal</span> : tool === "object" || tool === "move" ? <span className="canvas-context-label">{tool === "object" ? "Object selection" : "Pan image"}</span> : <div className="tool-group" aria-label="Selection tools">
          {(["brush", "eraser", "rectangle", "lasso"] as const).map((candidate) => (
            <button
              key={candidate}
              className={tool === candidate ? "selected" : ""}
              disabled={disabled}
              onClick={() => onToolChange(candidate)}
            >
              {toolLabel(candidate)}
            </button>
          ))}
        </div>}
        {!reviewing && (tool === "brush" || tool === "eraser") && (
          <label className="brush-control">
            <span>Brush</span>
            <input
              type="range"
              min="8"
              max="256"
              value={brushSize}
              onChange={(event) => setBrushSize(Number(event.target.value))}
            />
            <output>{brushSize}px</output>
          </label>
        )}
        {!reviewing && hasSelection && <details className="selection-details"><summary>Refine selection</summary><div className="selection-popover">
        <label className="edge-control">
          <span>Edge</span>
          <input
            type="range"
            min="1"
            max="64"
            value={edgeSize}
            onChange={(event) => setEdgeSize(Number(event.target.value))}
          />
          <output>{edgeSize}px</output>
        </label>
        <div className="tool-group mask-actions">
          <button disabled={disabled || !undoStackRef.current.length} onClick={undo}>Undo mask</button>
          <button disabled={disabled || !redoStackRef.current.length} onClick={redo}>Redo mask</button>
          <button disabled={disabled || !hasSelection} onClick={() => applyMaskTransform("grow")}>Grow</button>
          <button disabled={disabled || !hasSelection} onClick={() => applyMaskTransform("shrink")}>Shrink</button>
          <button disabled={disabled || !hasSelection} onClick={() => applyMaskTransform("feather")}>Feather</button>
          <button disabled={disabled || !imageSize} onClick={() => applyMaskTransform("invert")}>Invert</button>
          <button disabled={disabled || !hasSelection} onClick={clearMask}>Clear</button>
        </div></div></details>}
        <div className="zoom-controls">
          <button aria-label="Zoom out" onClick={() => changeZoom(zoom - 0.1)}>−</button>
          <button className={fitToStage ? "selected" : ""} onClick={() => setFitToStage(true)}>
            Fit
          </button>
          <button className={!fitToStage && manualZoom === 1 ? "selected" : ""} onClick={() => changeZoom(1)}>100%</button>
          <button aria-label="Zoom in" onClick={() => changeZoom(zoom + 0.1)}>+</button>
          <output>{Math.round(zoom * 100)}%</output>
        </div>
      </div>

      <div
        ref={stageRef}
        className="editor-stage"
        data-tool={tool}
        onPointerDown={beginPan}
        onPointerMove={updatePan}
        onPointerUp={finishPan}
        onPointerCancel={finishPan}
      >
        {imageUrl && imageSize && displaySize ? (
          <div className="editor-surface" style={{ width: displaySize.width, height: displaySize.height }}>
            <img src={previewImageUrl || imageUrl} alt={imageName || "Source image"} draggable={false} />
            <canvas
              ref={canvasRef}
              width={imageSize.width}
              height={imageSize.height}
              data-tool={tool}
              aria-label="Selection canvas"
              style={{ pointerEvents: disabled ? "none" : undefined, opacity: reviewing ? 0 : undefined }}
              onPointerDown={beginGesture}
              onPointerMove={updateGesture}
              onPointerUp={finishGesture}
              onPointerCancel={cancelGesture}
            />
            {!reviewing && points.map((point, index) => <i key={index} className={`selection-point ${point.label ? "positive" : "negative"}`} style={{ left: point.x * zoom, top: point.y * zoom }}>{point.label ? "+" : "−"}</i>)}
            {!reviewing && showContext && contextBox && <div className="context-boundary" aria-label="Protected context boundary" style={{ left: contextBox.x * zoom, top: contextBox.y * zoom, width: contextBox.width * zoom, height: contextBox.height * zoom }}><span>Context · stays unchanged</span></div>}
          </div>
        ) : (
          <div className="canvas-empty">
            <span className="empty-mark">+</span>
            <strong>Open, drop, or choose a recent image</strong>
            <span>Your mask stays aligned to the image at its native resolution.</span>
          </div>
        )}
      </div>

      <footer className="canvas-status" data-history-version={historyVersion}>
        <span>{imageSize ? `${imageSize.width} × ${imageSize.height} px` : "No document"}</span>
        <span>{reviewing ? "Apply to save this proposal" : toolStatus(tool)}</span>
      </footer>
    </section>
  );
}

function isMeaningfulGesture(gesture: PointerGesture): boolean {
  if (gesture.tool === "brush" || gesture.tool === "eraser") return gesture.points.length > 1;
  if (gesture.tool === "lasso") return gesture.points.length >= 3;
  const end = gesture.points[1];
  return Boolean(end && Math.abs(end.x - gesture.start.x) >= 1 && Math.abs(end.y - gesture.start.y) >= 1);
}

function toolLabel(tool: CanvasTool): string {
  return { object: "Object", brush: "Brush", eraser: "Eraser", rectangle: "Rectangle", lasso: "Lasso", move: "Pan" }[tool];
}

function toolStatus(tool: CanvasTool): string {
  if (tool === "object") return "Click an object · Alt + click to refine";
  if (tool === "move") return "Drag to pan · middle mouse works with any tool";
  if (tool === "brush") return "Paint your selection · hold Alt to subtract";
  if (tool === "eraser") return "Paint to erase selection";
  if (tool === "rectangle") return "Drag a rectangle · hold Alt to subtract";
  return "Draw around an area · hold Alt to subtract";
}

function trimHistory(stack: MaskSnapshot[]) {
  let bytes = stack.reduce((total, snapshot) => total + snapshot.bytes, 0);
  while (stack.length > 1 && bytes > MAX_HISTORY_BYTES) {
    const removed = stack.shift();
    bytes -= removed?.bytes ?? 0;
  }
}
