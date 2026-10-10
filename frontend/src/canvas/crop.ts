import { ImageSize } from "./coordinates";

/** The new canvas in picture pixels; it may reach past the picture, which leaves transparent space. */
export interface CropRect { x: number; y: number; width: number; height: number }
export type CropHandle = "move" | "n" | "s" | "e" | "w" | "ne" | "nw" | "se" | "sw";
export type CropRatio = "free" | "original" | "1:1" | "4:5" | "3:2" | "16:9" | "9:16";

export const MIN_CROP = 16;
export const MAX_CROP = 4096;
export const CROP_RATIOS: readonly CropRatio[] = ["free", "original", "1:1", "4:5", "3:2", "16:9", "9:16"];

export function fullCrop(size: ImageSize): CropRect {
  return { x: 0, y: 0, width: size.width, height: size.height };
}

/** Width over height for a ratio, or null when free. */
export function ratioValue(ratio: CropRatio, size: ImageSize): number | null {
  if (ratio === "free") return null;
  if (ratio === "original") return size.width / size.height;
  const [width, height] = ratio.split(":").map(Number);
  return width / height;
}

/** The largest centred crop of this ratio that stays inside the picture. */
export function cropForRatio(size: ImageSize, ratio: number): CropRect {
  const width = Math.round(Math.min(size.width, size.height * ratio));
  const height = Math.round(Math.min(size.height, width / ratio));
  return { x: Math.floor((size.width - width) / 2), y: Math.floor((size.height - height) / 2), width, height };
}

/** Fits a size to the ratio and the canvas limits, keeping the side that was asked for where possible. */
function fitted(width: number, height: number, ratio: number | null, keep: "width" | "height"): ImageSize {
  width = Math.max(MIN_CROP, width); height = Math.max(MIN_CROP, height);
  if (ratio) {
    if (keep === "width") height = width / ratio; else width = height * ratio;
  }
  const grow = Math.max(1, MIN_CROP / width, MIN_CROP / height);
  width *= grow; height *= grow;
  if (ratio) {
    const shrink = Math.min(1, MAX_CROP / width, MAX_CROP / height);
    width *= shrink; height *= shrink;
  }
  return { width: Math.min(MAX_CROP, Math.round(width)), height: Math.min(MAX_CROP, Math.round(height)) };
}

/** The crop after dragging a handle by (dx, dy) picture pixels; the opposite side stays put. */
export function dragCrop(start: CropRect, handle: CropHandle, dx: number, dy: number, ratio: number | null): CropRect {
  if (handle === "move") return { ...start, x: Math.round(start.x + dx), y: Math.round(start.y + dy) };
  const west = handle.includes("w"), east = handle.includes("e"), north = handle.includes("n"), south = handle.includes("s");
  const width = start.width + (east ? dx : west ? -dx : 0);
  const height = start.height + (south ? dy : north ? -dy : 0);
  // Corners follow the side that moved most; an edge sets its own side.
  const keep = east || west ? (north || south) && Math.abs(height / start.height - 1) > Math.abs(width / start.width - 1) ? "height" : "width" : "height";
  const size = fitted(width, height, ratio, keep);
  const x = west ? start.x + start.width - size.width : east ? start.x : Math.round(start.x + (start.width - size.width) / 2);
  const y = north ? start.y + start.height - size.height : south ? start.y : Math.round(start.y + (start.height - size.height) / 2);
  return { x, y, ...size };
}

/** The crop at a typed size, around the same centre. */
export function resizeCrop(rect: CropRect, width: number, height: number, ratio: number | null, keep: "width" | "height"): CropRect {
  const size = fitted(width, height, ratio, keep);
  return { x: Math.round(rect.x + (rect.width - size.width) / 2), y: Math.round(rect.y + (rect.height - size.height) / 2), ...size };
}

/** A crop the backend accepts: whole pixels, within the limits, touching the picture and not the picture as it is. */
export function validCrop(size: ImageSize, rect: CropRect): boolean {
  const values = [rect.x, rect.y, rect.width, rect.height];
  return values.every(Number.isInteger)
    && rect.width >= MIN_CROP && rect.height >= MIN_CROP && rect.width <= MAX_CROP && rect.height <= MAX_CROP
    && rect.x < size.width && rect.y < size.height && rect.x + rect.width > 0 && rect.y + rect.height > 0
    && !(rect.x === 0 && rect.y === 0 && rect.width === size.width && rect.height === size.height);
}

/** Whether the crop reaches past the picture, adding transparent space. */
export function addsSpace(size: ImageSize, rect: CropRect): boolean {
  return rect.x < 0 || rect.y < 0 || rect.x + rect.width > size.width || rect.y + rect.height > size.height;
}

/** What the crop view shows, in picture pixels: the picture with room around it, and the whole crop. */
export function cropBounds(size: ImageSize, rect: CropRect): CropRect {
  const room = Math.round(Math.max(size.width, size.height) * .12);
  const pad = Math.round(room / 3);
  const left = Math.min(-room, rect.x - pad), top = Math.min(-room, rect.y - pad);
  const right = Math.max(size.width + room, rect.x + rect.width + pad), bottom = Math.max(size.height + room, rect.y + rect.height + pad);
  return { x: left, y: top, width: right - left, height: bottom - top };
}
