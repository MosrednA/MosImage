export interface Point {
  x: number;
  y: number;
}

export interface ViewportRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface ImageSize {
  width: number;
  height: number;
}

export function viewportPointToImage(
  clientPoint: Point,
  viewport: ViewportRect,
  image: ImageSize,
): Point {
  if (viewport.width <= 0 || viewport.height <= 0) {
    return { x: 0, y: 0 };
  }

  return {
    x: clamp(((clientPoint.x - viewport.left) / viewport.width) * image.width, 0, image.width),
    y: clamp(((clientPoint.y - viewport.top) / viewport.height) * image.height, 0, image.height),
  };
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}
