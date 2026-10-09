import { PointerEvent as ReactPointerEvent, useEffect, useRef, useState } from "react";
import { maskBounds } from "./context";
import { ImageSize, Point, viewportPointToImage } from "./coordinates";

interface Options {
  imageUrl: string | null;
  imageSize: ImageSize | null;
  colour: string;
  size: number;
  erase: boolean;
  clearToken: number;
  onChange: (sketch: Blob | null) => void;
}

interface Stroke { pointerId: number; last: Point; erase: boolean }

const MAX_HISTORY_BYTES = 96 * 1024 * 1024;

/**
 * Rough coloured strokes over the picture, at its native resolution. They stay
 * a separate drawing until Render turns them into the real thing.
 */
export function useSketchLayer({ imageUrl, imageSize, colour, size, erase, clearToken, onChange }: Options) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const strokeRef = useRef<Stroke | null>(null);
  const undoRef = useRef<ImageData[]>([]);
  const versionRef = useRef(0);
  const [hasStrokes, setHasStrokes] = useState(false);
  const [canUndo, setCanUndo] = useState(false);

  // A new picture or Apply starts a fresh sketch.
  useEffect(() => {
    const canvas = canvasRef.current;
    canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
    undoRef.current = []; strokeRef.current = null; versionRef.current += 1;
    setCanUndo(false); setHasStrokes(false); onChange(null);
  }, [imageUrl, clearToken, onChange]);

  function context() { return canvasRef.current?.getContext("2d", { willReadFrequently: true }) ?? null; }

  function pointFor(event: ReactPointerEvent<HTMLCanvasElement>): Point | null {
    if (!imageSize) return null;
    return viewportPointToImage({ x: event.clientX, y: event.clientY }, event.currentTarget.getBoundingClientRect(), imageSize);
  }

  function remember() {
    const canvas = canvasRef.current, ctx = context();
    if (!canvas || !ctx) return;
    const stack = undoRef.current;
    stack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    let bytes = stack.reduce((total, item) => total + item.data.length, 0);
    while (stack.length > 1 && bytes > MAX_HISTORY_BYTES) bytes -= stack.shift()!.data.length;
    setCanUndo(true);
  }

  function paint(from: Point, to: Point, stroke: Stroke) {
    const ctx = context();
    if (!ctx) return;
    ctx.save();
    ctx.globalCompositeOperation = stroke.erase ? "destination-out" : "source-over";
    ctx.strokeStyle = ctx.fillStyle = colour;
    ctx.lineWidth = size; ctx.lineCap = "round"; ctx.lineJoin = "round";
    if (from.x === to.x && from.y === to.y) {
      ctx.beginPath(); ctx.arc(to.x, to.y, size / 2, 0, Math.PI * 2); ctx.fill();
    } else {
      ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y); ctx.stroke();
    }
    ctx.restore();
  }

  function begin(event: ReactPointerEvent<HTMLCanvasElement>) {
    const point = pointFor(event);
    if (!point || strokeRef.current) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    remember();
    const stroke = { pointerId: event.pointerId, last: point, erase: erase !== event.altKey };
    strokeRef.current = stroke;
    paint(point, point, stroke);
  }

  function update(event: ReactPointerEvent<HTMLCanvasElement>) {
    const stroke = strokeRef.current, point = pointFor(event);
    if (!stroke || stroke.pointerId !== event.pointerId || !point) return;
    paint(stroke.last, point, stroke);
    stroke.last = point;
  }

  function finish(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (strokeRef.current?.pointerId !== event.pointerId) return;
    strokeRef.current = null;
    emit();
  }

  function emit() {
    const canvas = canvasRef.current, ctx = context();
    if (!canvas || !ctx) return;
    const version = ++versionRef.current;
    const drawn = maskBounds(ctx.getImageData(0, 0, canvas.width, canvas.height)) !== null;
    setHasStrokes(drawn);
    if (!drawn) { onChange(null); return; }
    canvas.toBlob(blob => { if (version === versionRef.current) onChange(blob); }, "image/png");
  }

  function undo() {
    const previous = undoRef.current.pop();
    if (previous) context()?.putImageData(previous, 0, 0);
    setCanUndo(undoRef.current.length > 0);
    emit();
  }

  function clear() {
    const canvas = canvasRef.current;
    if (!canvas || !hasStrokes) return;
    remember();
    context()?.clearRect(0, 0, canvas.width, canvas.height);
    emit();
  }

  return { canvasRef, hasStrokes, canUndo, undo, clear, gesture: { begin, update, finish, cancel: finish } };
}
