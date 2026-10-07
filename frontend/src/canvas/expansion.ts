import { Expansion } from "../api";
import { ImageSize } from "./coordinates";

export function expansionForRatio(size: ImageSize, ratio: number): Expansion {
  const width = Math.max(size.width, Math.ceil(size.height * ratio));
  const height = Math.max(size.height, Math.ceil(size.width / ratio));
  const horizontal = width - size.width, vertical = height - size.height;
  return { left: Math.floor(horizontal / 2), right: Math.ceil(horizontal / 2), top: Math.floor(vertical / 2), bottom: Math.ceil(vertical / 2) };
}

export function expansionSize(size: ImageSize, expansion: Expansion): ImageSize {
  return { width: size.width + expansion.left + expansion.right, height: size.height + expansion.top + expansion.bottom };
}

export function validExpansion(size: ImageSize, expansion: Expansion): boolean {
  const target = expansionSize(size, expansion);
  return Object.values(expansion).every(value => Number.isInteger(value) && value >= 0 && value <= 2048)
    && Object.values(expansion).some(value => value > 0) && target.width <= 4096 && target.height <= 4096;
}
