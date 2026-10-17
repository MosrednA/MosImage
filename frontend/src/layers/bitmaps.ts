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

/**
 * Updates `editedData` in place so a layer edit keeps transparent areas cut out
 * while letting new details grow outside the original silhouette and erased parts
 * become transparent. The engine renders transparent pixels on `rgb(232, 232, 232)`.
 */
export function editedLayerPixels(editedData: Uint8ClampedArray, originalData: Uint8ClampedArray, width: number, height: number): void {
  const total = width * height;
  let hasTransparency = false;
  for (let i = 3; i < originalData.length; i += 4) {
    if (originalData[i] < 255) { hasTransparency = true; break; }
  }
  if (!hasTransparency) return;

  let bgR = 0, bgG = 0, bgB = 0, bgCount = 0;
  for (let p = 0; p < total; p += 1) {
    const idx = p * 4;
    if (originalData[idx + 3] === 0) {
      const r = editedData[idx], g = editedData[idx + 1], b = editedData[idx + 2];
      if (Math.abs(r - 232) <= 28 && Math.abs(g - 232) <= 28 && Math.abs(b - 232) <= 28) {
        bgR += r; bgG += g; bgB += b; bgCount += 1;
      }
    }
  }
  if (bgCount > 0) {
    bgR = Math.round(bgR / bgCount);
    bgG = Math.round(bgG / bgCount);
    bgB = Math.round(bgB / bgCount);
  } else {
    bgR = bgG = bgB = 232;
  }

  const bgDiffs = new Uint8Array(total);
  const changeDiffs = new Uint8Array(total);
  const candidate = new Uint8Array(total);
  for (let p = 0; p < total; p += 1) {
    const idx = p * 4;
    const r = editedData[idx], g = editedData[idx + 1], b = editedData[idx + 2];
    const bgDiff = Math.max(Math.abs(r - bgR), Math.abs(g - bgG), Math.abs(b - bgB));
    bgDiffs[p] = bgDiff;

    const origAlpha = originalData[idx + 3];
    const a0 = origAlpha / 255;
    const expR = originalData[idx] * a0 + bgR * (1 - a0);
    const expG = originalData[idx + 1] * a0 + bgG * (1 - a0);
    const expB = originalData[idx + 2] * a0 + bgB * (1 - a0);
    const changeDiff = Math.min(255, Math.round(Math.max(Math.abs(r - expR), Math.abs(g - expG), Math.abs(b - expB))));
    changeDiffs[p] = changeDiff;

    const origBgDiff = Math.max(Math.abs(originalData[idx] - bgR), Math.abs(originalData[idx + 1] - bgG), Math.abs(originalData[idx + 2] - bgB));
    if ((origAlpha < 240 && bgDiff <= 22) || (origAlpha >= 240 && origBgDiff > 28 && changeDiff > 24 && bgDiff <= 14)) {
      candidate[p] = 1;
    }
  }

  const exterior = new Uint8Array(total);
  const queue = new Int32Array(total);
  let head = 0, tail = 0;
  const push = (p: number) => {
    if (candidate[p] && !exterior[p]) {
      exterior[p] = 1;
      queue[tail++] = p;
    }
  };

  for (let x = 0; x < width; x += 1) {
    if (originalData[x * 4 + 3] < 128) push(x);
    const bottom = (height - 1) * width + x;
    if (originalData[bottom * 4 + 3] < 128) push(bottom);
  }
  for (let y = 0; y < height; y += 1) {
    const left = y * width;
    if (originalData[left * 4 + 3] < 128) push(left);
    const right = y * width + width - 1;
    if (originalData[right * 4 + 3] < 128) push(right);
  }

  // Also seed transparent holes inside the layer: gaps in its own content, such as
  // between strands of hair, and any of at least 16 connected pixels. A small grey
  // patch inside new detail only borders that detail, so it stays.
  const visitedHole = new Uint8Array(total);
  const component: number[] = [];
  for (let p = 0; p < total; p += 1) {
    if (visitedHole[p] || originalData[p * 4 + 3] !== 0 || bgDiffs[p] > 18) continue;
    component.length = 0;
    visitedHole[p] = 1;
    component.push(p);
    let bordersLayer = false;
    const visit = (n: number) => {
      if (originalData[n * 4 + 3] > 0) bordersLayer = true;
      else if (!visitedHole[n] && bgDiffs[n] <= 18) { visitedHole[n] = 1; component.push(n); }
    };
    for (let idx = 0; idx < component.length; idx += 1) {
      const cur = component[idx];
      const cx = cur % width, cy = (cur - cx) / width;
      if (cx > 0) visit(cur - 1);
      if (cx + 1 < width) visit(cur + 1);
      if (cy > 0) visit(cur - width);
      if (cy + 1 < height) visit(cur + width);
    }
    if (bordersLayer || component.length >= 16) {
      for (const node of component) push(node);
    }
  }

  while (head < tail) {
    const cur = queue[head++];
    const cx = cur % width, cy = (cur - cx) / width;
    if (cx > 0) push(cur - 1);
    if (cx + 1 < width) push(cur + 1);
    if (cy > 0) push(cur - width);
    if (cy + 1 < height) push(cur + width);
  }

  for (let p = 0; p < total; p += 1) {
    const idx = p * 4;
    const origAlpha = originalData[idx + 3];
    const bgDiff = bgDiffs[p];
    const changeDiff = changeDiffs[p];
    let alpha = 255;

    if (exterior[p]) {
      if (origAlpha > 0 && changeDiff <= 16) {
        alpha = origAlpha;
      } else if (bgDiff <= 16) {
        alpha = 0;
      } else {
        alpha = Math.round(((bgDiff - 16) / (22 - 16)) * 255);
      }
    } else if (origAlpha > 0 && origAlpha < 255 && fitsCoverage(editedData, idx, bgR, bgG, bgB, origAlpha)) {
      // A soft edge the edit kept or recoloured keeps its softness; only colour
      // that coverage cannot make over the background is new, solid detail.
      alpha = origAlpha;
    } else if (origAlpha < 255) {
      const cx = p % width, cy = (p - cx) / width;
      const touchesExterior =
        (cx > 0 && exterior[p - 1]) ||
        (cx + 1 < width && exterior[p + 1]) ||
        (cy > 0 && exterior[p - width]) ||
        (cy + 1 < height && exterior[p + width]);
      if (touchesExterior) {
        if (changeDiff <= 16) {
          alpha = origAlpha;
        } else if (bgDiff < 28) {
          const ramp = Math.round(Math.max(0, (bgDiff - 14) / (28 - 14)) * 255);
          alpha = Math.max(origAlpha, ramp);
        }
      }
    }

    editedData[idx + 3] = alpha;
    if (alpha > 0 && alpha < 255) {
      const a = alpha / 255;
      editedData[idx] = Math.min(255, Math.max(0, Math.round((editedData[idx] - bgR * (1 - a)) / a)));
      editedData[idx + 1] = Math.min(255, Math.max(0, Math.round((editedData[idx + 1] - bgG * (1 - a)) / a)));
      editedData[idx + 2] = Math.min(255, Math.max(0, Math.round((editedData[idx + 2] - bgB * (1 - a)) / a)));
    }
  }
}

/** True when an edited colour over the background is some colour at `alpha`, give or take 16 levels. */
function fitsCoverage(data: Uint8ClampedArray, idx: number, bgR: number, bgG: number, bgB: number, alpha: number): boolean {
  const a = alpha / 255;
  const fits = (value: number, bg: number) => value >= bg * (1 - a) - 16 && value <= bg * (1 - a) + 255 * a + 16;
  return fits(data[idx], bgR) && fits(data[idx + 1], bgG) && fits(data[idx + 2], bgB);
}

/**
 * Puts an edit of a masked layer back under its mask, in place, so the mask stays
 * editable: `editedData` is what editedLayerPixels made of the layer as the AI saw
 * it, through the alpha of `maskData`. Where the mask lets it show, it shows just
 * as edited; parts the AI never saw and left empty keep `layerData`, so painting
 * the mask back reveals them as before.
 */
export function unmaskedLayerPixels(editedData: Uint8ClampedArray, layerData: Uint8ClampedArray, maskData: Uint8ClampedArray): void {
  for (let idx = 0; idx < editedData.length; idx += 4) {
    const shown = maskData[idx + 3];
    if (shown === 255) continue;
    const alpha = editedData[idx + 3];
    if (alpha === 0 && layerData[idx + 3] * shown < 255) {
      editedData.set(layerData.subarray(idx, idx + 4), idx);
    } else if (shown > 0) {
      editedData[idx + 3] = Math.min(255, Math.round((alpha * 255) / shown));
    }
    // New detail where the mask hides everything is kept, hidden until the mask shows it.
  }
}

/**
 * A layer edit as the layer's new image under its unchanged `mask`: background stays
 * cut out, soft edges stay soft and new detail may grow past the old outline.
 */
export function editedLayerImage(edited: CanvasImageSource, image: CanvasImageSource, mask: HTMLCanvasElement | null, width: number, height: number): HTMLCanvasElement {
  const canvas = blankCanvas(width, height);
  const context = canvas.getContext("2d", { willReadFrequently: true })!;
  context.drawImage(edited, 0, 0, width, height);
  const editedPixels = context.getImageData(0, 0, width, height);
  // The AI saw the layer through its mask.
  editedLayerPixels(editedPixels.data, pixelData(maskedLayer(image, mask, width, height), width, height), width, height);
  if (mask) unmaskedLayerPixels(editedPixels.data, pixelData(image, width, height), pixelData(mask, width, height));
  context.putImageData(editedPixels, 0, 0);
  return canvas;
}

function pixelData(source: CanvasImageSource, width: number, height: number): Uint8ClampedArray {
  const canvas = blankCanvas(width, height);
  const context = canvas.getContext("2d", { willReadFrequently: true })!;
  context.drawImage(source, 0, 0, canvas.width, canvas.height);
  return context.getImageData(0, 0, canvas.width, canvas.height).data;
}

/**
 * Computes a feathered alpha mask over white for local edits between `sourceData`
 * and one or more `editedDataList` buffers. Returns `null` when the change covers
 * almost nothing (< 0.2%) or is global (> 45% of the image).
 */
export function changeMaskPixels(sourceData: Uint8ClampedArray, editedDataList: Uint8ClampedArray[], width: number, height: number): Uint8ClampedArray | null {
  const total = width * height;
  if (!total || !editedDataList.length) return null;
  const raw = new Uint8ClampedArray(total);
  for (let p = 0; p < total; p += 1) {
    const idx = p * 4;
    const sr = sourceData[idx], sg = sourceData[idx + 1], sb = sourceData[idx + 2];
    let maxDiff = 0;
    for (const editedData of editedDataList) {
      const diff = Math.max(Math.abs(editedData[idx] - sr), Math.abs(editedData[idx + 1] - sg), Math.abs(editedData[idx + 2] - sb));
      if (diff > maxDiff) maxDiff = diff;
    }
    raw[p] = maxDiff >= 24 ? 255 : 0;
  }

  const maxSide = Math.max(width, height);
  const openRadius = Math.max(1, Math.round(maxSide / 128));
  const opened = blurChannel(raw, width, height, openRadius);
  let activeCount = 0;
  for (let p = 0; p < total; p += 1) {
    const v = opened[p] > 120 ? 255 : 0;
    opened[p] = v;
    if (v > 0) activeCount += 1;
  }

  const coverage = activeCount / total;
  if (coverage < 0.002 || coverage > 0.45) return null;

  const growRadius = Math.max(1, Math.round(maxSide / 48));
  const grown = blurChannel(opened, width, height, growRadius).map(v => v > 24 ? 255 : 0);
  const featherRadius = Math.max(1, Math.round(maxSide / 40));
  const feathered = blurChannel(grown, width, height, featherRadius);

  const output = new Uint8ClampedArray(total * 4);
  for (let p = 0; p < total; p += 1) {
    const idx = p * 4;
    output[idx] = output[idx + 1] = output[idx + 2] = 255;
    output[idx + 3] = Math.max(opened[p], Math.min(255, Math.round(feathered[p] * 1.25)));
  }
  return output;
}

/**
 * Builds a full-size layer mask covering only the changed region of a whole-image
 * Edit so untouched background stays at native resolution. Returns `null` for global edits.
 */
export function editChangeMask(source: CanvasImageSource, edited: CanvasImageSource[], size: ImageSize): HTMLCanvasElement | null {
  if (!edited.length) return null;
  const maxWorkSide = 256;
  const scale = Math.min(1, maxWorkSide / Math.max(size.width, size.height));
  const workWidth = Math.max(1, Math.round(size.width * scale));
  const workHeight = Math.max(1, Math.round(size.height * scale));

  const readDownsampled = (image: CanvasImageSource): Uint8ClampedArray => {
    const c = blankCanvas(workWidth, workHeight);
    const ctx = c.getContext("2d", { willReadFrequently: true })!;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "medium";
    ctx.drawImage(image, 0, 0, workWidth, workHeight);
    return ctx.getImageData(0, 0, workWidth, workHeight).data;
  };

  const maskData = changeMaskPixels(readDownsampled(source), edited.map(readDownsampled), workWidth, workHeight);
  if (!maskData) return null;

  const workMask = blankCanvas(workWidth, workHeight);
  const workCtx = workMask.getContext("2d")!;
  const imageData = workCtx.createImageData(workWidth, workHeight);
  imageData.data.set(maskData);
  workCtx.putImageData(imageData, 0, 0);

  if (workWidth === size.width && workHeight === size.height) return workMask;
  const fullMask = blankCanvas(size.width, size.height);
  const fullCtx = fullMask.getContext("2d")!;
  fullCtx.imageSmoothingEnabled = true;
  fullCtx.imageSmoothingQuality = "high";
  fullCtx.drawImage(workMask, 0, 0, size.width, size.height);
  return fullMask;
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
