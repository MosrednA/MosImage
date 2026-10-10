/**
 * Gaussian blur in script, for WebKit (Safari and the macOS app), whose 2D
 * canvas has no `filter`. Three box blurs approximate the Gaussian that the
 * CSS `blur(sigma)` filter draws, and like it they treat outside pixels as empty.
 */

/** Whether this browser's 2D canvas applies CSS filters such as `blur()`. */
export const CANVAS_FILTER = typeof CanvasRenderingContext2D !== "undefined" && "filter" in CanvasRenderingContext2D.prototype;

/** Blurs one 8-bit channel, such as alpha, with a Gaussian of standard deviation `sigma` px. */
export function blurChannel(values: Uint8ClampedArray, width: number, height: number, sigma: number): Uint8ClampedArray {
  if (values.length !== width * height) throw new Error("Channel size does not match its dimensions.");
  const source = Float32Array.from(values);
  if (sigma > 0) {
    const scratch = new Float32Array(source.length);
    for (const size of boxSizes(sigma)) {
      const radius = (size - 1) / 2;
      if (!radius) continue;
      boxPass(source, scratch, width, height, radius, 1, width);
      boxPass(scratch, source, height, width, radius, width, 1);
    }
  }
  return Uint8ClampedArray.from(source);
}

/** Odd box widths whose three passes match a Gaussian of `sigma` (Kovesi's method). */
export function boxSizes(sigma: number, passes = 3): number[] {
  const ideal = Math.sqrt((12 * sigma * sigma) / passes + 1);
  let lower = Math.floor(ideal);
  if (lower % 2 === 0) lower -= 1;
  const lowerCount = Math.round((12 * sigma * sigma - passes * lower * lower - 4 * passes * lower - 3 * passes) / (-4 * lower - 4));
  return Array.from({ length: passes }, (_, index) => index < lowerCount ? lower : lower + 2);
}

/**
 * One running-sum box blur along `length` pixels `step` apart, for each of
 * `lines` lines that start `stride` apart.
 */
function boxPass(source: Float32Array, target: Float32Array, length: number, lines: number, radius: number, step: number, stride: number): void {
  const scale = 1 / (2 * radius + 1);
  for (let line = 0; line < lines; line += 1) {
    const start = line * stride;
    let sum = 0;
    for (let index = 0; index <= radius && index < length; index += 1) sum += source[start + index * step];
    for (let index = 0; index < length; index += 1) {
      target[start + index * step] = sum * scale;
      const entering = index + radius + 1;
      if (entering < length) sum += source[start + entering * step];
      const leaving = index - radius;
      if (leaving >= 0) sum -= source[start + leaving * step];
    }
  }
}
