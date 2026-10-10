import { PointerEvent as ReactPointerEvent, useEffect, useRef, useState } from "react";
import { isTyping } from "../hooks/useWindowEvents";
import { invertMaskPixels } from "../masks/maskOperations";
import { EdgeSettings, MaskField, alphaBounds, overlayPixels, refineAlpha } from "../masks/maskRefine";
import { signedDistanceFieldAsync } from "../masks/fieldWorker";
import { ImageBox, maskBounds } from "./context";
import { ImageSize, Point, viewportPointToImage } from "./coordinates";
import { MaskTool } from "./tools";

type SelectionOperation = "add" | "subtract";

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
  edges: EdgeSettings;
  zoom: number;
  clearToken: number;
  maskImportUrl: string | null;
  onMaskChange: (mask: Blob | null) => void;
  onError: (message: string) => void;
}

const MAX_HISTORY_BYTES = 96 * 1024 * 1024;
const OVERLAY_COLOR: [number, number, number] = [96, 165, 255];
const EXPORT_DELAY = 120;

interface RefinedMask { box: ImageBox; alpha: Uint8ClampedArray }

/**
 * The editable selection. The drawing canvas holds the painted selection at
 * native resolution, with undo history and gestures. Expand/Soften are applied
 * non-destructively on top; the overlay canvas shows exactly the mask that is
 * sent for inpainting.
 */
export function useMaskLayer({ imageUrl, imageSize, disabled, brushSize, edges, zoom, clearToken, maskImportUrl, onMaskChange, onError }: Options) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const fieldRef = useRef<MaskField | null>(null);
  /** Only the newest painted selection may set the field. */
  const fieldRequestRef = useRef(0);
  const refinedRef = useRef<RefinedMask | null>(null);
  const [fieldVersion, setFieldVersion] = useState(0);
  const [drawing, setDrawing] = useState(false);
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
    fieldRequestRef.current += 1;
    fieldRef.current = null;
    refinedRef.current = null;
    setHasSelection(false);
    setSelectionBounds(null);
    onMaskChange(null);
  }, [imageUrl, onMaskChange]);

  useEffect(() => {
    if (!imageSize) return;
    for (const canvas of [canvasRef.current, overlayRef.current]) {
      if (!canvas) continue;
      canvas.width = imageSize.width;
      canvas.height = imageSize.height;
      canvas.getContext("2d")?.clearRect(0, 0, imageSize.width, imageSize.height);
    }
  }, [imageSize]);

  // Re-apply the edge settings whenever the painted selection or a slider changes.
  useEffect(() => {
    const field = fieldRef.current;
    if (!field) {
      refinedRef.current = null;
      maskVersionRef.current += 1;
      setSelectionBounds(null);
      drawOverlay();
      onMaskChange(null);
      return;
    }
    const alpha = refineAlpha(field, edges);
    const refined = { box: field.box, alpha };
    refinedRef.current = refined;
    const bounds = alphaBounds(alpha, field.box);
    setSelectionBounds(bounds);
    drawOverlay();
    const version = ++maskVersionRef.current;
    if (!bounds) { onMaskChange(null); return; }
    const timer = window.setTimeout(() => exportMask(refined, version), EXPORT_DELAY);
    return () => window.clearTimeout(timer);
  }, [fieldVersion, edges.expand, edges.soften]);

  useEffect(() => {
    const frame = window.requestAnimationFrame(drawOverlay);
    return () => window.cancelAnimationFrame(frame);
  }, [zoom]);

  useEffect(() => {
    if (!clearToken) return;
    clear();
  }, [clearToken]);

  useEffect(() => {
    const handleUndo = (event: KeyboardEvent) => {
      if (disabled || !(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== "z") return;
      if (isTyping(event)) return;
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

  function begin(event: ReactPointerEvent<HTMLCanvasElement>, tool: MaskTool, subtract = false) {
    const point = pointFor(event);
    const base = rememberMask();
    if (!point || !base) return;
    const gesture: PointerGesture = {
      pointerId: event.pointerId,
      tool,
      operation: tool === "eraser" || subtract || event.altKey ? "subtract" : "add",
      start: point,
      points: [point],
      base,
    };
    gestureRef.current = gesture;
    setDrawing(true);
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
    setDrawing(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    emitMask();
  }

  function cancel(event: ReactPointerEvent<HTMLCanvasElement>) {
    const gesture = gestureRef.current;
    if (!gesture || gesture.pointerId !== event.pointerId) return;
    context2d()?.putImageData(gesture.base, 0, 0);
    undoStackRef.current.pop();
    gestureRef.current = null;
    setDrawing(false);
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

  function invert() {
    const canvas = canvasRef.current;
    const context = context2d();
    const source = rememberMask();
    if (!canvas || !context || !source) return;
    const result = context.createImageData(canvas.width, canvas.height);
    result.data.set(invertMaskPixels({ width: canvas.width, height: canvas.height, data: source.data }));
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

  /** The painted selection changed: rebuild its distance field, off the main thread. */
  function emitMask() {
    const canvas = canvasRef.current;
    const context = context2d();
    if (!canvas || !context) return;
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
    const bounds = maskBounds(pixels);
    const request = ++fieldRequestRef.current;
    setHasSelection(bounds !== null);
    if (!bounds) { fieldRef.current = null; setFieldVersion(version => version + 1); return; }
    void signedDistanceFieldAsync(pixels.data, canvas.width, canvas.height, bounds).then(field => {
      // A newer stroke or another picture replaces this one.
      if (request !== fieldRequestRef.current) return;
      fieldRef.current = field;
      setFieldVersion(version => version + 1);
    });
  }

  function exportMask(refined: RefinedMask, version: number) {
    if (!imageSize) return;
    const { box, alpha } = refined;
    const output = document.createElement("canvas");
    output.width = imageSize.width;
    output.height = imageSize.height;
    const rgba = new ImageData(box.width, box.height);
    for (let index = 0; index < alpha.length; index += 1) {
      const offset = index * 4;
      rgba.data[offset] = rgba.data[offset + 1] = rgba.data[offset + 2] = 255;
      rgba.data[offset + 3] = alpha[index];
    }
    output.getContext("2d")?.putImageData(rgba, box.x, box.y);
    output.toBlob(blob => { if (version === maskVersionRef.current) onMaskChange(blob); }, "image/png");
  }

  function drawOverlay() {
    const overlay = overlayRef.current;
    const context = overlay?.getContext("2d");
    if (!overlay || !context) return;
    context.clearRect(0, 0, overlay.width, overlay.height);
    const field = fieldRef.current, refined = refinedRef.current;
    if (!field || !refined) return;
    const image = context.createImageData(field.box.width, field.box.height);
    image.data.set(overlayPixels(field, edges, refined.alpha, Math.max(1, 1.5 / zoom), OVERLAY_COLOR));
    context.putImageData(image, field.box.x, field.box.y);
  }

  return {
    canvasRef,
    overlayRef,
    drawing,
    hasSelection,
    selectionBounds,
    canUndo: undoStackRef.current.length > 0,
    canRedo: redoStackRef.current.length > 0,
    undo, redo, clear, invert,
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
