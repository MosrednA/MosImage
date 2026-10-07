import { PointerEvent as ReactPointerEvent, useEffect, useRef, useState } from "react";
import { featherMaskPixels, growMaskPixels, invertMaskPixels, shrinkMaskPixels } from "../masks/maskOperations";
import { ImageBox, maskBounds } from "./context";
import { ImageSize, Point, viewportPointToImage } from "./coordinates";
import { MaskTool } from "./tools";

type SelectionOperation = "add" | "subtract";
export type MaskTransform = "invert" | "grow" | "shrink" | "feather";

interface MaskSnapshot { bytes: number; imageData: ImageData }

interface PointerGesture {
  pointerId: number;
  tool: MaskTool;
  operation: SelectionOperation;
  start: Point;
  points: Point[];
  base: ImageData;
}

interface Options {
  imageUrl: string | null;
  imageSize: ImageSize | null;
  disabled: boolean;
  brushSize: number;
  edgeSize: number;
  clearToken: number;
  maskImportUrl: string | null;
  onMaskChange: (mask: Blob | null) => void;
  onError: (message: string) => void;
}

const MAX_HISTORY_BYTES = 96 * 1024 * 1024;

/** The editable selection: a native-resolution alpha canvas with undo history and drawing gestures. */
export function useMaskLayer({ imageUrl, imageSize, disabled, brushSize, edgeSize, clearToken, maskImportUrl, onMaskChange, onError }: Options) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const gestureRef = useRef<PointerGesture | null>(null);
  const undoStackRef = useRef<MaskSnapshot[]>([]);
  const redoStackRef = useRef<MaskSnapshot[]>([]);
  const maskVersionRef = useRef(0);
  const [hasSelection, setHasSelection] = useState(false);
  const [selectionBounds, setSelectionBounds] = useState<ImageBox | null>(null);
  const [, setHistoryVersion] = useState(0);
  const touchHistory = () => setHistoryVersion(version => version + 1);

  useEffect(() => {
    maskVersionRef.current += 1;
    gestureRef.current = null;
    undoStackRef.current = [];
    redoStackRef.current = [];
    touchHistory();
    setHasSelection(false);
    setSelectionBounds(null);
    onMaskChange(null);
  }, [imageUrl, onMaskChange]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!imageSize || !canvas) return;
    canvas.width = imageSize.width;
    canvas.height = imageSize.height;
    canvas.getContext("2d")?.clearRect(0, 0, imageSize.width, imageSize.height);
  }, [imageSize]);

  useEffect(() => {
    if (!clearToken) return;
    clear();
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
    const failed = () => { if (alive) onError("Could not load the selection. Select the object again."); };
    void fetch(maskImportUrl).then(response => {
      if (!response.ok) throw new Error("Could not load selection");
      return response.blob();
    }).then(blob => {
      if (!alive) return;
      url = URL.createObjectURL(blob);
      const image = new Image();
      image.onload = () => {
        if (!alive) return;
        const canvas = canvasRef.current;
        const context = canvas?.getContext("2d", { willReadFrequently: true });
        if (!canvas || !context || image.width !== canvas.width || image.height !== canvas.height) {
          onError("The selection does not match the current image. Select the object again.");
          return;
        }
        rememberMask();
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.drawImage(image, 0, 0);
        const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
        // Selection masks arrive as greyscale; the editor stores selection in alpha.
        for (let i = 0; i < pixels.data.length; i += 4) {
          pixels.data[i + 3] = pixels.data[i];
          pixels.data[i] = pixels.data[i + 1] = pixels.data[i + 2] = 255;
        }
        context.putImageData(pixels, 0, 0);
        emitMask();
      };
      image.onerror = failed;
      image.src = url;
    }).catch(() => { if (alive) { onMaskChange(null); failed(); } });
    return () => { alive = false; if (url) URL.revokeObjectURL(url); };
  }, [maskImportUrl, imageSize]);

  function context2d() {
    return canvasRef.current?.getContext("2d", { willReadFrequently: true }) ?? null;
  }

  function captureMask(): ImageData | null {
    const canvas = canvasRef.current;
    const context = context2d();
    if (!canvas || !context) return null;
    return context.getImageData(0, 0, canvas.width, canvas.height);
  }

  function rememberMask(): ImageData | null {
    const imageData = captureMask();
    if (!imageData) return null;
    undoStackRef.current.push({ imageData, bytes: imageData.data.byteLength });
    trimHistory(undoStackRef.current);
    redoStackRef.current = [];
    touchHistory();
    return imageData;
  }

  function pointFor(event: ReactPointerEvent<HTMLCanvasElement>): Point | null {
    const canvas = canvasRef.current;
    if (!canvas || !imageSize) return null;
    return viewportPointToImage({ x: event.clientX, y: event.clientY }, canvas.getBoundingClientRect(), imageSize);
  }

  function begin(event: ReactPointerEvent<HTMLCanvasElement>, tool: MaskTool) {
    const point = pointFor(event);
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

  function update(event: ReactPointerEvent<HTMLCanvasElement>) {
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

  function finish(event: ReactPointerEvent<HTMLCanvasElement>) {
    const gesture = gestureRef.current;
    if (!gesture || gesture.pointerId !== event.pointerId) return;
    update(event);
    if (!isMeaningfulGesture(gesture)) {
      context2d()?.putImageData(gesture.base, 0, 0);
      undoStackRef.current.pop();
      touchHistory();
    }
    gestureRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    emitMask();
  }

  function cancel(event: ReactPointerEvent<HTMLCanvasElement>) {
    const gesture = gestureRef.current;
    if (!gesture || gesture.pointerId !== event.pointerId) return;
    context2d()?.putImageData(gesture.base, 0, 0);
    undoStackRef.current.pop();
    gestureRef.current = null;
    touchHistory();
    emitMask();
  }

  function drawBrushSegment(gesture: PointerGesture, point: Point) {
    const context = context2d();
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
    const context = context2d();
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

  function transform(kind: MaskTransform) {
    const canvas = canvasRef.current;
    const context = context2d();
    const source = rememberMask();
    if (!canvas || !context || !source) return;
    const pixels = { width: canvas.width, height: canvas.height, data: source.data };
    const transformed = kind === "invert" ? invertMaskPixels(pixels)
      : kind === "grow" ? growMaskPixels(pixels, edgeSize)
        : kind === "shrink" ? shrinkMaskPixels(pixels, edgeSize)
          : featherMaskPixels(pixels, edgeSize);
    const result = context.createImageData(canvas.width, canvas.height);
    result.data.set(transformed);
    context.putImageData(result, 0, 0);
    emitMask();
  }

  function clear() {
    const canvas = canvasRef.current;
    const context = context2d();
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
    context2d()?.putImageData(snapshot.imageData, 0, 0);
    touchHistory();
    emitMask();
  }

  function emitMask() {
    const canvas = canvasRef.current;
    const context = context2d();
    if (!canvas || !context) return;
    const bounds = maskBounds(context.getImageData(0, 0, canvas.width, canvas.height));
    setSelectionBounds(bounds);
    setHasSelection(bounds !== null);
    const version = ++maskVersionRef.current;
    if (!bounds) { onMaskChange(null); return; }
    canvas.toBlob(blob => { if (version === maskVersionRef.current) onMaskChange(blob); }, "image/png");
  }

  return {
    canvasRef,
    hasSelection,
    selectionBounds,
    canUndo: undoStackRef.current.length > 0,
    canRedo: redoStackRef.current.length > 0,
    undo, redo, clear, transform,
    gesture: { begin, update, finish, cancel },
  };
}

function isMeaningfulGesture(gesture: PointerGesture): boolean {
  if (gesture.tool === "brush" || gesture.tool === "eraser") return gesture.points.length > 1;
  if (gesture.tool === "lasso") return gesture.points.length >= 3;
  const end = gesture.points[1];
  return Boolean(end && Math.abs(end.x - gesture.start.x) >= 1 && Math.abs(end.y - gesture.start.y) >= 1);
}

function trimHistory(stack: MaskSnapshot[]) {
  let bytes = stack.reduce((total, snapshot) => total + snapshot.bytes, 0);
  while (stack.length > 1 && bytes > MAX_HISTORY_BYTES) {
    const removed = stack.shift();
    bytes -= removed?.bytes ?? 0;
  }
}
