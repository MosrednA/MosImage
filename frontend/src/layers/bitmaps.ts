import { LayerRecord, fileUrl } from "../api";
import { ImageBox, maskBounds } from "../canvas/context";
import { ImageSize } from "../canvas/coordinates";
import { CANVAS_FILTER, blurChannel } from "./blur";
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

/**
 * Decodes an image file exactly, ready to draw. WebKit's createImageBitmap
 * dithers (black mask pixels come back as 1, so a thresholded selection
 * covers the whole canvas); an <img> decodes exactly in every engine, and
 * keeps drawing after its object URL is revoked.
 */
export async function decodeImage(file: Blob): Promise<HTMLImageElement> {
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    return image;
  } finally {
    URL.revokeObjectURL(url);
  }
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

/** A layer's mask as it applies now: none while it is switched off. */
export function shownMask(layer: Pick<LayerRecord, "mask_url" | "mask_enabled">): Promise<HTMLCanvasElement | null> {
  return layer.mask_enabled && layer.mask_url ? loadMask(fileUrl(layer.mask_url)) : Promise.resolve(null);
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

/** Draws the layer image through its mask; any image's alpha can serve as the mask. */
export function composeLayer(target: HTMLCanvasElement, image: CanvasImageSource, mask: CanvasImageSource | null): void {
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

/** How a layer is drawn: as it shows, its mask alone in grey (Alt-click), or with hidden parts tinted red (\). */
export type LayerLook = "layer" | "mask" | "overlay";
const HIDDEN_TINT = "rgba(255, 40, 40, 0.5)";

/** Draws a layer in one of its looks; `mask` is its mask whether or not it is switched on. */
export function renderLayer(target: HTMLCanvasElement, image: CanvasImageSource, mask: CanvasImageSource | null, look: LayerLook): void {
  if (look === "layer") { composeLayer(target, image, mask); return; }
  const context = target.getContext("2d");
  if (!context) return;
  const { width, height } = target;
  context.save();
  if (look === "mask") {
    // White shows, black hides, like a Photoshop mask on its own.
    context.globalCompositeOperation = "copy";
    context.fillStyle = mask ? "black" : "white";
    context.fillRect(0, 0, width, height);
    context.globalCompositeOperation = "source-over";
    if (mask) context.drawImage(mask, 0, 0, width, height);
  } else {
    composeLayer(target, image, mask);
    if (mask) {
      const tint = blankCanvas(width, height);
      const tinted = tint.getContext("2d")!;
      tinted.fillStyle = HIDDEN_TINT;
      tinted.fillRect(0, 0, width, height);
      tinted.globalCompositeOperation = "destination-out";
      tinted.drawImage(mask, 0, 0, width, height);
      context.drawImage(tint, 0, 0);
    }
  }
  context.restore();
}

/** A selection on the picture (coverage in alpha) as a mask in a layer's own pixels. */
export async function selectionToLayerMask(selection: Blob, size: ImageSize, layer: Placement): Promise<HTMLCanvasElement> {
  const bitmap = await decodeImage(selection);
  const mask = blankCanvas(layer.width, layer.height);
  const context = mask.getContext("2d")!;
  // The inverse of the layer's placement: canvas pixels into layer pixels.
  context.translate(layer.width / 2, layer.height / 2);
  context.scale(1 / layer.scale_x, 1 / layer.scale_y);
  context.rotate((-layer.rotation * Math.PI) / 180);
  context.translate(-layer.x, -layer.y);
  context.drawImage(bitmap, 0, 0, size.width, size.height);
  return mask;
}

/** A layer's image through its mask, at its own size: what an edit of that layer alone sees. */
export function maskedLayer(image: CanvasImageSource, mask: HTMLCanvasElement | null, width: number, height: number): HTMLCanvasElement {
  const canvas = blankCanvas(width, height);
  composeLayer(canvas, image, mask);
  return canvas;
}

/** An edited layer image cut back to the original's transparency, so cut-outs stay cut out. */
export function withTransparencyOf(edited: CanvasImageSource, original: CanvasImageSource, width: number, height: number): HTMLCanvasElement {
  const canvas = blankCanvas(width, height);
  composeLayer(canvas, edited, original);
  return canvas;
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

/**
 * Layers drawn as one, as Merge does: cut to `box` on the canvas and made
 * smaller when a side would pass `maxSide` px, as PNG.
 */
export async function mergeLayers(parts: FlatLayer[], box: ImageBox, maxSide: number): Promise<Blob> {
  const factor = Math.min(1, maxSide / Math.max(box.width, box.height));
  const canvas = blankCanvas(box.width * factor, box.height * factor);
  const context = canvas.getContext("2d")!;
  context.imageSmoothingQuality = "high";
  context.scale(canvas.width / box.width, canvas.height / box.height);
  context.translate(-box.x, -box.y);
  for (const layer of parts) drawLayer(context, layer, layer.opacity);
  return canvasBlob(canvas);
}

/**
 * Where a layer shows on the canvas, soft edges and all, as a greyscale
 * selection mask: Ctrl-click on a layer thumbnail in Photoshop. Null when
 * none of it is on the canvas.
 */
export async function layerSelection(size: ImageSize, layer: FlatLayer): Promise<Blob | null> {
  const drawn = blankCanvas(size.width, size.height);
  const context = drawn.getContext("2d", { willReadFrequently: true })!;
  drawLayer(context, layer, 1);
  const alpha = context.getImageData(0, 0, drawn.width, drawn.height).data;
  let shows = false;
  for (let index = 3; index < alpha.length && !shows; index += 4) shows = alpha[index] > 0;
  if (!shows) return null;
  context.globalCompositeOperation = "source-in";
  context.fillStyle = "white";
  context.fillRect(0, 0, drawn.width, drawn.height);
  const selection = blankCanvas(size.width, size.height);
  const output = selection.getContext("2d")!;
  output.fillStyle = "black";
  output.fillRect(0, 0, selection.width, selection.height);
  output.drawImage(drawn, 0, 0);
  return canvasBlob(selection);
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
  const drawn = blankCanvas(size.width, size.height);
  // Read back in script where the canvas cannot blur; a draw source otherwise.
  drawLayer(drawn.getContext("2d", CANVAS_FILTER ? undefined : { willReadFrequently: true })!, layer, 1);
  if (!CANVAS_FILTER) {
    const context = drawn.getContext("2d")!;
    const pixels = context.getImageData(0, 0, size.width, size.height);
    footprintPixels(pixels.data, size.width, size.height, grow, soften);
    context.putImageData(pixels, 0, 0);
    return canvasBlob(drawn);
  }
  const shape = blankCanvas(size.width, size.height);
  const context = shape.getContext("2d", { willReadFrequently: true })!;
  // Blurring and then keeping every touched pixel grows the shape by about the blur radius.
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

/**
 * `layerFootprint` without canvas filters, for WebKit: turns RGBA layer pixels
 * in place into the grown and softened greyscale footprint of their alpha.
 */
export function footprintPixels(data: Uint8ClampedArray, width: number, height: number, grow: number, soften: number): void {
  const alpha = new Uint8ClampedArray(width * height);
  for (let pixel = 0; pixel < alpha.length; pixel += 1) alpha[pixel] = data[pixel * 4 + 3];
  const grown = blurChannel(alpha, width, height, grow / 2).map(value => value > 8 ? 255 : 0);
  const footprint = blurChannel(grown, width, height, soften);
  for (let pixel = 0; pixel < footprint.length; pixel += 1) {
    data[pixel * 4] = data[pixel * 4 + 1] = data[pixel * 4 + 2] = footprint[pixel];
    data[pixel * 4 + 3] = 255;
  }
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
  const mask = await decodeImage(selection);
  const maskCanvas = blankCanvas(size.width, size.height);
  const maskContext = maskCanvas.getContext("2d", { willReadFrequently: true })!;
  maskContext.drawImage(mask, 0, 0, size.width, size.height);
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
  const mask = await decodeImage(selection);
  const maskCanvas = blankCanvas(size.width, size.height);
  const maskContext = maskCanvas.getContext("2d", { willReadFrequently: true })!;
  maskContext.drawImage(mask, 0, 0, size.width, size.height);
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

export async function fileFromUrl(url: string, name: string): Promise<File> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error("Could not open the saved image.");
  const blob = await response.blob();
  return new File([blob], name, { type: blob.type || "image/png" });
}
