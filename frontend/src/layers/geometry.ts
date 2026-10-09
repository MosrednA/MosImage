import { Point } from "../canvas/coordinates";

/** Where a layer sits: centre in canvas pixels, natural size, scale (negative flips) and rotation in degrees. */
export interface Placement {
  x: number;
  y: number;
  width: number;
  height: number;
  scale_x: number;
  scale_y: number;
  rotation: number;
}

export type Handle = "n" | "s" | "e" | "w" | "ne" | "nw" | "se" | "sw";
export const HANDLES: readonly Handle[] = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

/** Handle positions in layer space, 0–1 across the layer. */
const HANDLE_UNIT: Record<Handle, Point> = {
  nw: { x: 0, y: 0 }, n: { x: 0.5, y: 0 }, ne: { x: 1, y: 0 }, e: { x: 1, y: 0.5 },
  se: { x: 1, y: 1 }, s: { x: 0.5, y: 1 }, sw: { x: 0, y: 1 }, w: { x: 0, y: 0.5 },
};
const MIN_SCALE = 0.01;
const ROTATION_SNAP = 15;

/** A layer pixel (origin top-left) in canvas pixels. */
export function layerToCanvas(layer: Placement, point: Point): Point {
  const local = { x: (point.x - layer.width / 2) * layer.scale_x, y: (point.y - layer.height / 2) * layer.scale_y };
  const turned = rotate(local, layer.rotation);
  return { x: layer.x + turned.x, y: layer.y + turned.y };
}

/** A canvas pixel in layer pixels; outside 0–width/0–height when it misses the layer. */
export function canvasToLayer(layer: Placement, point: Point): Point {
  const local = rotate({ x: point.x - layer.x, y: point.y - layer.y }, -layer.rotation);
  return { x: local.x / layer.scale_x + layer.width / 2, y: local.y / layer.scale_y + layer.height / 2 };
}

export function containsPoint(layer: Placement, point: Point): boolean {
  const local = canvasToLayer(layer, point);
  return local.x >= 0 && local.y >= 0 && local.x <= layer.width && local.y <= layer.height;
}

export function handlePoint(layer: Placement, handle: Handle): Point {
  const unit = HANDLE_UNIT[handle];
  return layerToCanvas(layer, { x: unit.x * layer.width, y: unit.y * layer.height });
}

/** The layer's corners on the canvas, clockwise from its top-left. */
export function corners(layer: Placement): Point[] {
  return (["nw", "ne", "se", "sw"] as const).map(handle => handlePoint(layer, handle));
}

/** Moves by the pointer's travel since the drag started. */
export function moved<T extends Placement>(start: T, from: Point, to: Point): T {
  return { ...start, x: start.x + to.x - from.x, y: start.y + to.y - from.y };
}

/**
 * Resizes from a handle while the opposite handle stays put. Corners keep the
 * aspect ratio unless `free`; edges stretch one side, or both with `free`.
 */
export function resized<T extends Placement>(start: T, handle: Handle, pointer: Point, free: boolean): T {
  const unit = HANDLE_UNIT[handle];
  const anchorUnit = { x: 1 - unit.x, y: 1 - unit.y };
  const anchor = layerToCanvas(start, { x: anchorUnit.x * start.width, y: anchorUnit.y * start.height });
  // Pointer and handle relative to the anchor, in the layer's unrotated frame.
  const reach = rotate({ x: pointer.x - anchor.x, y: pointer.y - anchor.y }, -start.rotation);
  const span = { x: (unit.x - anchorUnit.x) * start.width * start.scale_x, y: (unit.y - anchorUnit.y) * start.height * start.scale_y };
  const corner = span.x !== 0 && span.y !== 0;
  let factorX = span.x !== 0 ? reach.x / span.x : 1;
  let factorY = span.y !== 0 ? reach.y / span.y : 1;
  if (corner && !free) {
    factorX = factorY = (reach.x * span.x + reach.y * span.y) / (span.x * span.x + span.y * span.y);
  } else if (!corner && free) {
    if (span.x !== 0) factorY = factorX; else factorX = factorY;
  }
  // Dragging past the anchor stops at the minimum size; flipping has its own buttons.
  factorX = Math.max(factorX, MIN_SCALE / Math.abs(start.scale_x));
  factorY = Math.max(factorY, MIN_SCALE / Math.abs(start.scale_y));
  const scale_x = start.scale_x * factorX, scale_y = start.scale_y * factorY;
  const offset = rotate({ x: (anchorUnit.x - 0.5) * start.width * scale_x, y: (anchorUnit.y - 0.5) * start.height * scale_y }, start.rotation);
  return { ...start, scale_x, scale_y, x: anchor.x - offset.x, y: anchor.y - offset.y };
}

/** Turns around the centre by the pointer's sweep; `snap` rounds to 15°. */
export function rotated<T extends Placement>(start: T, from: Point, to: Point, snap: boolean): T {
  const sweep = degrees(Math.atan2(to.y - start.y, to.x - start.x) - Math.atan2(from.y - start.y, from.x - start.x));
  const rotation = start.rotation + sweep;
  return { ...start, rotation: normalizeAngle(snap ? Math.round(rotation / ROTATION_SNAP) * ROTATION_SNAP : rotation) };
}

/** Centred on the canvas, scaled down to fit within 80% of it. */
export function initialPlacement(size: { width: number; height: number }, canvas: { width: number; height: number }) {
  const scale = Math.min(1, (canvas.width * 0.8) / size.width, (canvas.height * 0.8) / size.height);
  return { x: canvas.width / 2, y: canvas.height / 2, scale_x: scale, scale_y: scale, rotation: 0 };
}

/** A full generation fills as much of the canvas as its aspect ratio allows. */
export function generationPlacement(size: { width: number; height: number }, canvas: { width: number; height: number }) {
  const scale = Math.min(canvas.width / size.width, canvas.height / size.height);
  return { x: canvas.width / 2, y: canvas.height / 2, scale_x: scale, scale_y: scale, rotation: 0 };
}

/** CSS transform for an element of the layer's natural size at the top-left of a surface shown at `zoom`. */
export function cssTransform(layer: Placement, zoom: number): string {
  return `translate(${layer.x * zoom}px, ${layer.y * zoom}px) rotate(${layer.rotation}deg) scale(${layer.scale_x * zoom}, ${layer.scale_y * zoom}) translate(${-layer.width / 2}px, ${-layer.height / 2}px)`;
}

export function normalizeAngle(angle: number): number {
  const turned = ((angle % 360) + 360) % 360;
  return turned > 180 ? turned - 360 : turned;
}

function rotate(point: Point, angle: number): Point {
  const radians = (angle * Math.PI) / 180, cos = Math.cos(radians), sin = Math.sin(radians);
  return { x: point.x * cos - point.y * sin, y: point.x * sin + point.y * cos };
}

function degrees(radians: number): number {
  return (radians * 180) / Math.PI;
}
