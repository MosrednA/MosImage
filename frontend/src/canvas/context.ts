import { MaskPixels } from "../masks/maskOperations";
import { ImageSize } from "./coordinates";

export interface ImageBox { x: number; y: number; width: number; height: number }

export function maskBounds(mask: MaskPixels): ImageBox | null {
  let left = mask.width;
  let top = mask.height;
  let right = 0;
  let bottom = 0;
  for (let pixel = 0; pixel < mask.width * mask.height; pixel += 1) {
    if (!mask.data[pixel * 4 + 3]) continue;
    const x = pixel % mask.width;
    const y = Math.floor(pixel / mask.width);
    left = Math.min(left, x); top = Math.min(top, y);
    right = Math.max(right, x + 1); bottom = Math.max(bottom, y + 1);
  }
  return right > left && bottom > top ? { x: left, y: top, width: right - left, height: bottom - top } : null;
}

export function paddedContextBox(selection: ImageBox, image: ImageSize, padding: number): ImageBox {
  const x = Math.max(0, selection.x - padding);
  const y = Math.max(0, selection.y - padding);
  const right = Math.min(image.width, selection.x + selection.width + padding);
  const bottom = Math.min(image.height, selection.y + selection.height + padding);
  return { x, y, width: right - x, height: bottom - y };
}
