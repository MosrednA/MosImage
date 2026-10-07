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
