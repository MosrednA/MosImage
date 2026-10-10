import { ImageBox } from "../canvas/context";

/**
 * Non-destructive selection edges.
 *
 * The painted selection is converted once into a signed distance field
 * (positive inside, negative outside, in native pixels). Expand and Soften are
 * then cheap per-pixel mappings, so their sliders can preview live.
 *
 * Soften ramps *outward* from the expanded edge: the whole selection is
 * regenerated and the transition blends into the surrounding original. An
 * inward-only ramp kept original pixels at the selection edge, leaving the
 * outline of a removed object visible.
 */
export interface MaskField {
  box: ImageBox;
  signed: Float32Array;
}

export interface EdgeSettings {
  expand: number;
  soften: number;
}

export const EXPAND_RANGE = { min: -32, max: 64 } as const;
export const SOFTEN_RANGE = { min: 0, max: 96 } as const;
export const DEFAULT_EXPAND = 4;

/**
 * Roughly the long side / 200. A selection drawn or found by SAM often stops a
 * few pixels inside the object; the old outline then stays, like a blue rim
 * around a recoloured shirt. The error grows with the picture.
 */
export function autoExpand(width: number, height: number): number {
  return Math.round(Math.min(Math.max(Math.max(width, height) / 200, DEFAULT_EXPAND), 24));
}

/**
 * Roughly the long side / 150. The engine already matches colour and grain at
 * the edge; a wider ramp half-blends new shapes over the original, such as a
 * new neckline smeared over skin.
 */
export function autoSoften(width: number, height: number): number {
  return Math.round(Math.min(Math.max(Math.max(width, height) / 150, 6), 32));
}

/** Region of the image that any edge setting can affect. */
export function fieldMargin(): number {
  return EXPAND_RANGE.max + SOFTEN_RANGE.max + 2;
}

/** Signed chamfer distance for the RGBA alpha mask, limited to `bounds` padded by `margin`. */
export function signedDistanceField(rgba: Uint8ClampedArray, imageWidth: number, imageHeight: number, bounds: ImageBox, margin = fieldMargin()): MaskField {
  const crop = cropSelection(rgba, imageWidth, imageHeight, bounds, margin);
  return distanceField(crop.box, crop.selected);
}

/** The selected pixels (alpha ≥ 128) within `bounds` padded by `margin`: the cheap first half of the field. */
export function cropSelection(rgba: Uint8ClampedArray, imageWidth: number, imageHeight: number, bounds: ImageBox, margin = fieldMargin()): { box: ImageBox; selected: Uint8Array } {
  const x0 = Math.max(0, bounds.x - margin);
  const y0 = Math.max(0, bounds.y - margin);
  const x1 = Math.min(imageWidth, bounds.x + bounds.width + margin);
  const y1 = Math.min(imageHeight, bounds.y + bounds.height + margin);
  const width = x1 - x0, height = y1 - y0;
  const selected = new Uint8Array(width * height);
  for (let y = 0; y < height; y += 1) {
    const row = ((y + y0) * imageWidth + x0) * 4 + 3;
    for (let x = 0; x < width; x += 1) selected[y * width + x] = rgba[row + x * 4] >= 128 ? 1 : 0;
  }
  return { box: { x: x0, y: y0, width, height }, selected };
}

/** Signed distance for a cropped selection: the expensive half, safe to run in a worker. */
export function distanceField(box: ImageBox, selected: Uint8Array): MaskField {
  const count = box.width * box.height;
  const inside = new Float32Array(count);
  const outside = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    inside[index] = selected[index] ? Infinity : 0;
    outside[index] = selected[index] ? 0 : Infinity;
  }
  chamfer(inside, box.width, box.height);
  chamfer(outside, box.width, box.height);
  const signed = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    // Pixel centres sit half a pixel from the boundary between classes.
    signed[index] = selected[index] ? Math.min(inside[index], 1e6) - 0.5 : 0.5 - Math.min(outside[index], 1e6);
  }
  return { box, signed };
}

/** Alpha (0–255) for every pixel of the field's box. */
export function refineAlpha(field: MaskField, { expand, soften }: EdgeSettings): Uint8ClampedArray {
  const alpha = new Uint8ClampedArray(field.signed.length);
  for (let index = 0; index < alpha.length; index += 1) alpha[index] = edgeAlpha(field.signed[index] + expand, soften);
  return alpha;
}

export function edgeAlpha(distance: number, soften: number): number {
  if (distance > 0) return 255;
  if (soften <= 0) return 0;
  const t = 1 + distance / soften;
  if (t <= 0) return 0;
  return Math.round(255 * t * t * (3 - 2 * t));
}

/** Tight bounds of the non-zero alpha, in image coordinates. */
export function alphaBounds(alpha: Uint8ClampedArray, box: ImageBox): ImageBox | null {
  let left = box.width, top = box.height, right = -1, bottom = -1;
  for (let y = 0; y < box.height; y += 1) {
    for (let x = 0; x < box.width; x += 1) {
      if (!alpha[y * box.width + x]) continue;
      if (x < left) left = x;
      if (x > right) right = x;
      if (y < top) top = y;
      bottom = y;
    }
  }
  return right < 0 ? null : { x: box.x + left, y: box.y + top, width: right - left + 1, height: bottom - top + 1 };
}

/**
 * Overlay pixels: a translucent tint whose strength follows the alpha, with a
 * crisp line where full regeneration starts and a faint line where blending ends.
 */
export function overlayPixels(field: MaskField, edges: EdgeSettings, alpha: Uint8ClampedArray, lineWidth: number, color: [number, number, number]): Uint8ClampedArray {
  const output = new Uint8ClampedArray(alpha.length * 4);
  const [red, green, blue] = color;
  for (let index = 0; index < alpha.length; index += 1) {
    const distance = field.signed[index] + edges.expand;
    const offset = index * 4;
    let opacity = alpha[index] * 0.45;
    let r = red, g = green, b = blue;
    if (distance > 0 && distance <= lineWidth) { opacity = 255; r = g = b = 255; }
    else if (edges.soften > 0 && distance <= -edges.soften && distance > -edges.soften - lineWidth) { opacity = 150; }
    output[offset] = r; output[offset + 1] = g; output[offset + 2] = b; output[offset + 3] = opacity;
  }
  return output;
}

function chamfer(distance: Float32Array, width: number, height: number) {
  const diagonal = Math.SQRT2;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      let value = distance[index];
      if (value === 0) continue;
      if (x > 0) value = Math.min(value, distance[index - 1] + 1);
      if (y > 0) {
        const above = index - width;
        value = Math.min(value, distance[above] + 1);
        if (x > 0) value = Math.min(value, distance[above - 1] + diagonal);
        if (x < width - 1) value = Math.min(value, distance[above + 1] + diagonal);
      }
      distance[index] = value;
    }
  }
  for (let y = height - 1; y >= 0; y -= 1) {
    for (let x = width - 1; x >= 0; x -= 1) {
      const index = y * width + x;
      let value = distance[index];
      if (value === 0) continue;
      if (x < width - 1) value = Math.min(value, distance[index + 1] + 1);
      if (y < height - 1) {
        const below = index + width;
        value = Math.min(value, distance[below] + 1);
        if (x < width - 1) value = Math.min(value, distance[below + 1] + diagonal);
        if (x > 0) value = Math.min(value, distance[below - 1] + diagonal);
      }
      distance[index] = value;
    }
  }
}
