import { ImageBox } from "../canvas/context";
import { Point } from "../canvas/coordinates";

/** A round mask brush: radius in layer pixels, hardness 0 (soft) to 1 (hard). */
export interface BrushTip { radius: number; hardness: number }

/** Softness is the inverse of hardness: 0 cuts cleanly, 100 fades over the whole radius. */
export const SOFTNESS_RANGE = { min: 0, max: 100, initial: 50 } as const;
export const OPACITY_RANGE = { min: 5, max: 100, initial: 100 } as const;
/** The grey the brush paints on a mask: 100 is white and shows the layer, 0 is black and hides it. */
export const TONE_RANGE = { min: 0, max: 100, initial: 100 } as const;

/** Dabs a tenth of the radius apart keep soft edges smooth and hard edges round. */
const SPACING = 0.1;

/**
 * Stamps dabs from `from` to `to` (inclusive) onto a stroke canvas and returns
 * the area touched. Dabs combine by their maximum, not by piling up, so a soft
 * brush keeps its soft edge along the whole stroke, as in Photoshop.
 */
export function stampSegment(context: CanvasRenderingContext2D, from: Point, to: Point, tip: BrushTip): ImageBox {
  const distance = Math.hypot(to.x - from.x, to.y - from.y);
  const steps = distance ? Math.max(1, Math.ceil(distance / Math.max(0.5, tip.radius * SPACING))) : 0;
  context.save();
  context.globalCompositeOperation = "lighten";
  for (let step = steps ? 1 : 0; step <= steps; step += 1) {
    const t = steps ? step / steps : 0;
    stamp(context, { x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t }, tip);
  }
  context.restore();
  const reach = tip.radius + 2;
  const left = Math.floor(Math.min(from.x, to.x) - reach), top = Math.floor(Math.min(from.y, to.y) - reach);
  return clampBox({ x: left, y: top, width: Math.ceil(Math.max(from.x, to.x) + reach) - left, height: Math.ceil(Math.max(from.y, to.y) + reach) - top }, context.canvas);
}

/** One opaque dab: white in its hard core, easing to black at its rim, so `lighten` keeps the strongest. */
function stamp(context: CanvasRenderingContext2D, at: Point, tip: BrushTip) {
  context.beginPath();
  context.arc(at.x, at.y, tip.radius, 0, Math.PI * 2);
  if (tip.hardness >= 0.99) context.fillStyle = "white";
  else {
    const gradient = context.createRadialGradient(at.x, at.y, 0, at.x, at.y, tip.radius);
    const core = Math.max(0, Math.min(tip.hardness, 0.99));
    gradient.addColorStop(0, "white");
    gradient.addColorStop(core, "white");
    // Smoothstep falloff from the core to the rim.
    for (const t of [0.25, 0.5, 0.75]) gradient.addColorStop(core + (1 - core) * t, grey(1 - t * t * (3 - 2 * t)));
    gradient.addColorStop(1, "black");
    context.fillStyle = gradient;
  }
  context.fill();
}

/**
 * Applies a stroke to a mask inside `box`: the mask from before the stroke,
 * moved toward `tone` (1 shows, 0 hides) by the stroke at `opacity`, as
 * painting grey on a Photoshop mask does. Masks keep their coverage in alpha over white.
 */
export function applyStroke(mask: HTMLCanvasElement, base: HTMLCanvasElement, stroke: HTMLCanvasElement, box: ImageBox, tone: number, opacity: number): void {
  if (box.width <= 0 || box.height <= 0) return;
  const from = base.getContext("2d", { willReadFrequently: true })!.getImageData(box.x, box.y, box.width, box.height);
  const strength = stroke.getContext("2d", { willReadFrequently: true })!.getImageData(box.x, box.y, box.width, box.height);
  mixStroke(from.data, strength.data, tone, opacity);
  mask.getContext("2d")!.putImageData(from, box.x, box.y);
}

/** `coverage` (mask RGBA) takes the stroke (grey, in RGBA) in place: the pure part of applyStroke. */
export function mixStroke(coverage: Uint8ClampedArray, stroke: Uint8ClampedArray, tone: number, opacity: number): void {
  for (let index = 0; index < coverage.length; index += 4) {
    const amount = (stroke[index] * stroke[index + 3] * opacity) / (255 * 255);
    const alpha = coverage[index + 3] / 255;
    const next = alpha + (tone - alpha) * amount;
    coverage[index] = coverage[index + 1] = coverage[index + 2] = 255;
    coverage[index + 3] = Math.round(next * 255);
  }
}

function clampBox(box: ImageBox, canvas: { width: number; height: number }): ImageBox {
  const x = Math.max(0, box.x), y = Math.max(0, box.y);
  return { x, y, width: Math.max(0, Math.min(canvas.width, box.x + box.width) - x), height: Math.max(0, Math.min(canvas.height, box.y + box.height) - y) };
}

function grey(level: number): string {
  const value = Math.round(level * 255);
  return `rgb(${value}, ${value}, ${value})`;
}
