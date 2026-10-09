import { blankCanvas, canvasBlob } from "../layers/bitmaps";
import { refineAlpha, signedDistanceField } from "../masks/maskRefine";
import { ImageBox, maskBounds } from "./context";
import { ImageSize } from "./coordinates";

/** Plain colours for rough shapes: the AI reads them as the colours of the real thing. */
export const SKETCH_COLOURS = ["#1d1d1f", "#ffffff", "#d8392e", "#f08a24", "#f4d03f", "#3d9a4b", "#3878d6", "#7a4a2a", "#e6b593", "#9a5bc4"] as const;
export const SKETCH_SIZE = { min: 2, max: 160, initial: 18 } as const;

/** How far the rendered area reaches past the strokes: room for edges, shadows and a slightly larger result. */
export function sketchReach(size: ImageSize): number {
  return Math.round(Math.min(Math.max(Math.max(size.width, size.height) * 0.025, 8), 64));
}

/** Soft coverage around the strokes: fully redrawn within `reach`, then blending out over another `reach`. */
export function sketchFootprint(rgba: Uint8ClampedArray, size: ImageSize, reach: number): { box: ImageBox; alpha: Uint8ClampedArray } | null {
  const bounds = maskBounds({ data: rgba, width: size.width, height: size.height });
  if (!bounds) return null;
  const field = signedDistanceField(rgba, size.width, size.height, bounds, reach * 2 + 2);
  return { box: field.box, alpha: refineAlpha(field, { expand: reach, soften: reach }) };
}

/** The sketch's footprint as a selection mask (white, coverage in alpha), like the selection tools make. */
export async function sketchMask(sketch: Blob, size: ImageSize): Promise<Blob | null> {
  const strokes = await pixels(sketch, size);
  const footprint = sketchFootprint(strokes, size, sketchReach(size));
  if (!footprint) return null;
  const { box, alpha } = footprint;
  const rgba = new ImageData(box.width, box.height);
  for (let index = 0; index < alpha.length; index += 1) {
    rgba.data.fill(255, index * 4, index * 4 + 3);
    rgba.data[index * 4 + 3] = alpha[index];
  }
  const canvas = blankCanvas(size.width, size.height);
  canvas.getContext("2d")!.putImageData(rgba, box.x, box.y);
  return canvasBlob(canvas);
}

/** The picture with the sketch painted on it, as the AI should see it. */
export async function sketchedPicture(picture: Blob, sketch: Blob, size: ImageSize): Promise<File> {
  const canvas = blankCanvas(size.width, size.height);
  const context = canvas.getContext("2d")!;
  for (const layer of [picture, sketch]) {
    const bitmap = await createImageBitmap(layer);
    context.drawImage(bitmap, 0, 0, size.width, size.height);
    bitmap.close();
  }
  return new File([await canvasBlob(canvas)], "sketch.png", { type: "image/png" });
}

async function pixels(image: Blob, size: ImageSize): Promise<Uint8ClampedArray> {
  const canvas = blankCanvas(size.width, size.height);
  const context = canvas.getContext("2d", { willReadFrequently: true })!;
  const bitmap = await createImageBitmap(image);
  context.drawImage(bitmap, 0, 0, size.width, size.height);
  bitmap.close();
  return context.getImageData(0, 0, size.width, size.height).data;
}
