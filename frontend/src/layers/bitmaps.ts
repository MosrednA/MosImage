import { ImageBox, maskBounds } from "../canvas/context";
import { ImageSize } from "../canvas/coordinates";
import { Placement } from "./geometry";

const images = new Map<string, Promise<HTMLImageElement>>();
const masks = new Map<string, Promise<HTMLCanvasElement>>();

/** Loads an image once per URL; layer files never change after upload. */
export function loadImage(url: string): Promise<HTMLImageElement> {
  let loading = images.get(url);
  if (!loading) {
    loading = new Promise((resolve, reject) => {
      const image = new Image();
      image.crossOrigin = "anonymous";
      image.onload = () => resolve(image);
      image.onerror = () => { images.delete(url); reject(new Error("Could not load a layer image.")); };
      image.src = url;
    });
    images.set(url, loading);
  }
  return loading;
}

/** A stored greyscale layer mask as an editable canvas that keeps coverage in alpha. */
export function loadMask(url: string): Promise<HTMLCanvasElement> {
  let loading = masks.get(url);
  if (!loading) {
    loading = loadImage(url).then(image => {
      const canvas = blankCanvas(image.naturalWidth, image.naturalHeight);
      const context = canvas.getContext("2d", { willReadFrequently: true })!;
      context.drawImage(image, 0, 0);
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
      luminanceToAlpha(pixels.data);
      context.putImageData(pixels, 0, 0);
      return canvas;
    });
    masks.set(url, loading);
  }
  return loading.then(copyCanvas);
}

/** Moves greyscale coverage into alpha over white, the way the editor stores masks. */
export function luminanceToAlpha(data: Uint8ClampedArray): void {
  for (let index = 0; index < data.length; index += 4) {
    data[index + 3] = data[index];
    data[index] = data[index + 1] = data[index + 2] = 255;
  }
}

/** A fully visible mask for a layer that has none yet. */
export function solidMask(width: number, height: number): HTMLCanvasElement {
  const canvas = blankCanvas(width, height);
  const context = canvas.getContext("2d")!;
  context.fillStyle = "white";
  context.fillRect(0, 0, width, height);
  return canvas;
}

/** Draws the layer image through its mask. */
export function composeLayer(target: HTMLCanvasElement, image: CanvasImageSource, mask: HTMLCanvasElement | null): void {
  const context = target.getContext("2d");
  if (!context) return;
  context.save();
  context.globalCompositeOperation = "copy";
  context.drawImage(image, 0, 0, target.width, target.height);
  if (mask) {
    context.globalCompositeOperation = "destination-in";
    context.drawImage(mask, 0, 0, target.width, target.height);
  }
  context.restore();
}

export interface FlatLayer extends Placement { image: CanvasImageSource; mask: HTMLCanvasElement | null; opacity: number }

/** The background with the visible layers on top, cropped to the canvas, as PNG. */
export async function flatten(background: CanvasImageSource, size: ImageSize, layers: FlatLayer[]): Promise<Blob> {
  const canvas = blankCanvas(size.width, size.height);
  const context = canvas.getContext("2d")!;
  context.drawImage(background, 0, 0, size.width, size.height);
  for (const layer of layers) drawLayer(context, layer, layer.opacity);
  return canvasBlob(canvas);
}

/** Draws a layer through its mask at its place on the canvas. */
function drawLayer(context: CanvasRenderingContext2D, layer: FlatLayer, opacity: number): void {
  const composed = blankCanvas(layer.width, layer.height);
  composeLayer(composed, layer.image, layer.mask);
  context.save();
  context.globalAlpha = opacity;
  context.translate(layer.x, layer.y);
  context.rotate((layer.rotation * Math.PI) / 180);
  context.scale(layer.scale_x, layer.scale_y);
  context.drawImage(composed, -layer.width / 2, -layer.height / 2);
  context.restore();
}

/**
 * Where a layer shows on the canvas, grown by `grow` px and softened by
 * `soften` px, as a greyscale selection mask: what "Blend in" lets the AI redo.
 */
export async function layerFootprint(size: ImageSize, layer: FlatLayer, grow: number, soften: number): Promise<Blob> {
  const shape = blankCanvas(size.width, size.height);
  const context = shape.getContext("2d", { willReadFrequently: true })!;
  // Blurring and then keeping every touched pixel grows the shape by about the blur radius.
  const drawn = blankCanvas(size.width, size.height);
  drawLayer(drawn.getContext("2d")!, layer, 1);
  context.filter = `blur(${grow / 2}px)`;
  context.drawImage(drawn, 0, 0);
  context.filter = "none";
  const pixels = context.getImageData(0, 0, size.width, size.height);
  thresholdAlpha(pixels.data, 8);
  context.putImageData(pixels, 0, 0);

  const selection = blankCanvas(size.width, size.height);
  const output = selection.getContext("2d")!;
  output.fillStyle = "black";
  output.fillRect(0, 0, size.width, size.height);
  output.filter = `blur(${soften}px)`;
  output.drawImage(shape, 0, 0);
  return canvasBlob(selection);
}

/** Makes every pixel with more than `cutoff` alpha fully white and the rest transparent. */
export function thresholdAlpha(data: Uint8ClampedArray, cutoff: number): void {
  for (let index = 0; index < data.length; index += 4) {
    data[index] = data[index + 1] = data[index + 2] = 255;
    data[index + 3] = data[index + 3] > cutoff ? 255 : 0;
  }
}

export interface Patch { image: Blob; mask: Blob; box: ImageBox }

/**
 * The selected part of an AI result as a layer: the result cropped to the
 * selection. The API result already contains the feathered blend, so the
 * layer mask keeps its footprint without applying that blend a second time.
 */
export async function selectionPatch(result: CanvasImageSource, size: ImageSize, selection: Blob): Promise<Patch | null> {
  const mask = await createImageBitmap(selection);
  const maskCanvas = blankCanvas(size.width, size.height);
  const maskContext = maskCanvas.getContext("2d", { willReadFrequently: true })!;
  maskContext.drawImage(mask, 0, 0, size.width, size.height);
  mask.close();
  const coverage = maskContext.getImageData(0, 0, size.width, size.height);
  // AI selections are opaque greyscale PNGs, not alpha masks.
  luminanceToAlpha(coverage.data);
  thresholdAlpha(coverage.data, 0);
  maskContext.putImageData(coverage, 0, 0);
  const box = maskBounds(coverage);
  if (!box) return null;
  const image = blankCanvas(box.width, box.height);
  image.getContext("2d")!.drawImage(result, box.x, box.y, box.width, box.height, 0, 0, box.width, box.height);
  const cropped = blankCanvas(box.width, box.height);
  cropped.getContext("2d")!.drawImage(maskCanvas, box.x, box.y, box.width, box.height, 0, 0, box.width, box.height);
  const [imageBlob, maskBlob] = await Promise.all([canvasBlob(image), canvasBlob(cropped)]);
  return { image: imageBlob, mask: maskBlob, box };
}

export interface Cutout { blob: Blob; box: ImageBox }

/** The selected part of the background as a transparent PNG, cropped to the selection. */
export async function cutSelection(background: CanvasImageSource, size: ImageSize, selection: Blob): Promise<Cutout | null> {
  const mask = await createImageBitmap(selection);
  const maskCanvas = blankCanvas(size.width, size.height);
  const maskContext = maskCanvas.getContext("2d", { willReadFrequently: true })!;
  maskContext.drawImage(mask, 0, 0, size.width, size.height);
  mask.close();
  const box = maskBounds(maskContext.getImageData(0, 0, size.width, size.height));
  if (!box) return null;
  const cut = blankCanvas(box.width, box.height);
  const context = cut.getContext("2d")!;
  context.drawImage(background, box.x, box.y, box.width, box.height, 0, 0, box.width, box.height);
  context.globalCompositeOperation = "destination-in";
  context.drawImage(maskCanvas, box.x, box.y, box.width, box.height, 0, 0, box.width, box.height);
  return { blob: await canvasBlob(cut), box };
}

/** True when a composed layer canvas is visible enough at a layer pixel to count as a click on it. */
export function solidAt(canvas: HTMLCanvasElement, x: number, y: number): boolean {
  if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return false;
  const pixel = canvas.getContext("2d", { willReadFrequently: true })?.getImageData(Math.floor(x), Math.floor(y), 1, 1).data;
  return !!pixel && pixel[3] > 24;
}

export function canvasBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error("Could not prepare the image.")), "image/png"));
}

export function blankCanvas(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(width));
  canvas.height = Math.max(1, Math.round(height));
  return canvas;
}

export function copyCanvas(source: HTMLCanvasElement): HTMLCanvasElement {
  const canvas = blankCanvas(source.width, source.height);
  canvas.getContext("2d")!.drawImage(source, 0, 0);
  return canvas;
}

/** Turns hidden mask pixels visible and visible ones hidden. */
export function invertMask(mask: HTMLCanvasElement): HTMLCanvasElement {
  const canvas = blankCanvas(mask.width, mask.height);
  const context = canvas.getContext("2d")!;
  context.fillStyle = "white";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.globalCompositeOperation = "destination-out";
  context.drawImage(mask, 0, 0);
  return canvas;
}
