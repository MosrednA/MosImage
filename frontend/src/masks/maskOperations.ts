export interface MaskPixels {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

export function invertMaskPixels(source: MaskPixels): Uint8ClampedArray {
  const alpha = readAlpha(source);
  for (let index = 0; index < alpha.length; index += 1) {
    alpha[index] = 255 - alpha[index];
  }
  return whiteMaskWithAlpha(alpha);
}

export function growMaskPixels(source: MaskPixels, radius: number): Uint8ClampedArray {
  return whiteMaskWithAlpha(morphAlpha(readAlpha(source), source.width, source.height, radius, "maximum"));
}

export function shrinkMaskPixels(source: MaskPixels, radius: number): Uint8ClampedArray {
  return whiteMaskWithAlpha(morphAlpha(readAlpha(source), source.width, source.height, radius, "minimum"));
}

export function featherMaskPixels(source: MaskPixels, radius: number): Uint8ClampedArray {
  const alpha = readAlpha(source);
  const blurred = blurAlpha(alpha, source.width, source.height, radius);
  for (let pixel = 0; pixel < alpha.length; pixel += 1) blurred[pixel] = Math.min(alpha[pixel], blurred[pixel]);
  return whiteMaskWithAlpha(blurred);
}

function readAlpha(source: MaskPixels): Uint8ClampedArray {
  const expectedLength = source.width * source.height * 4;
  if (source.width <= 0 || source.height <= 0 || source.data.length !== expectedLength) {
    throw new Error("Mask pixel dimensions do not match the RGBA buffer.");
  }

  const alpha = new Uint8ClampedArray(source.width * source.height);
  for (let pixel = 0; pixel < alpha.length; pixel += 1) {
    alpha[pixel] = source.data[pixel * 4 + 3];
  }
  return alpha;
}

function whiteMaskWithAlpha(alpha: Uint8ClampedArray): Uint8ClampedArray {
  const output = new Uint8ClampedArray(alpha.length * 4);
  for (let pixel = 0; pixel < alpha.length; pixel += 1) {
    const offset = pixel * 4;
    output[offset] = 255;
    output[offset + 1] = 255;
    output[offset + 2] = 255;
    output[offset + 3] = alpha[pixel];
  }
  return output;
}

type Extrema = "maximum" | "minimum";

function morphAlpha(
  alpha: Uint8ClampedArray,
  width: number,
  height: number,
  requestedRadius: number,
  extrema: Extrema,
): Uint8ClampedArray {
  const radius = normalizeRadius(requestedRadius);
  if (radius === 0) return alpha.slice();

  const horizontal = new Uint8ClampedArray(alpha.length);
  for (let y = 0; y < height; y += 1) {
    const line = alpha.subarray(y * width, (y + 1) * width);
    horizontal.set(slidingExtrema(line, radius, extrema), y * width);
  }

  const output = new Uint8ClampedArray(alpha.length);
  const column = new Uint8ClampedArray(height);
  for (let x = 0; x < width; x += 1) {
    for (let y = 0; y < height; y += 1) column[y] = horizontal[y * width + x];
    const transformed = slidingExtrema(column, radius, extrema);
    for (let y = 0; y < height; y += 1) output[y * width + x] = transformed[y];
  }
  return output;
}

function slidingExtrema(values: Uint8ClampedArray, radius: number, extrema: Extrema): Uint8ClampedArray {
  const windowSize = radius * 2 + 1;
  const padded = new Uint8ClampedArray(values.length + radius * 2);
  padded.set(values, radius);
  const output = new Uint8ClampedArray(values.length);
  const deque = new Int32Array(padded.length);
  let head = 0;
  let tail = 0;

  for (let index = 0; index < padded.length; index += 1) {
    while (head < tail && deque[head] <= index - windowSize) head += 1;
    while (head < tail) {
      const previous = padded[deque[tail - 1]];
      const candidate = padded[index];
      if (extrema === "maximum" ? previous > candidate : previous < candidate) break;
      tail -= 1;
    }
    deque[tail] = index;
    tail += 1;

    const outputIndex = index - windowSize + 1;
    if (outputIndex >= 0 && outputIndex < output.length) {
      output[outputIndex] = padded[deque[head]];
    }
  }
  return output;
}

function blurAlpha(
  alpha: Uint8ClampedArray,
  width: number,
  height: number,
  requestedRadius: number,
): Uint8ClampedArray {
  const radius = normalizeRadius(requestedRadius);
  if (radius === 0) return alpha.slice();

  const horizontal = new Uint8ClampedArray(alpha.length);
  for (let y = 0; y < height; y += 1) {
    horizontal.set(boxBlurLine(alpha.subarray(y * width, (y + 1) * width), radius), y * width);
  }

  const output = new Uint8ClampedArray(alpha.length);
  const column = new Uint8ClampedArray(height);
  for (let x = 0; x < width; x += 1) {
    for (let y = 0; y < height; y += 1) column[y] = horizontal[y * width + x];
    const blurred = boxBlurLine(column, radius);
    for (let y = 0; y < height; y += 1) output[y * width + x] = blurred[y];
  }
  return output;
}

function boxBlurLine(values: Uint8ClampedArray, radius: number): Uint8ClampedArray {
  const windowSize = radius * 2 + 1;
  const padded = new Uint8ClampedArray(values.length + radius * 2);
  padded.set(values, radius);
  const output = new Uint8ClampedArray(values.length);
  let sum = 0;

  for (let index = 0; index < padded.length; index += 1) {
    sum += padded[index];
    if (index >= windowSize) sum -= padded[index - windowSize];
    const outputIndex = index - windowSize + 1;
    if (outputIndex >= 0 && outputIndex < output.length) output[outputIndex] = Math.round(sum / windowSize);
  }
  return output;
}

function normalizeRadius(radius: number): number {
  if (!Number.isFinite(radius)) throw new Error("Mask operation radius must be finite.");
  return Math.min(Math.max(Math.floor(radius), 0), 64);
}
