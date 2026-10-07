import { describe, expect, it } from "vitest";
import { featherMaskPixels, growMaskPixels, invertMaskPixels, shrinkMaskPixels } from "./maskOperations";

function mask(width: number, height: number, selected: Array<[number, number]> = []) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (const [x, y] of selected) {
    const offset = (y * width + x) * 4;
    data[offset] = 255;
    data[offset + 1] = 255;
    data[offset + 2] = 255;
    data[offset + 3] = 255;
  }
  return { width, height, data };
}

function alphaAt(data: Uint8ClampedArray, width: number, x: number, y: number) {
  return data[(y * width + x) * 4 + 3];
}

describe("mask pixel operations", () => {
  it("inverts alpha while keeping a white mask", () => {
    const result = invertMaskPixels(mask(2, 1, [[0, 0]]));
    expect(Array.from(result)).toEqual([255, 255, 255, 0, 255, 255, 255, 255]);
  });

  it("grows a selection without transposing a non-square mask", () => {
    const result = growMaskPixels(mask(5, 3, [[2, 1]]), 1);
    expect(alphaAt(result, 5, 1, 0)).toBe(255);
    expect(alphaAt(result, 5, 3, 2)).toBe(255);
    expect(alphaAt(result, 5, 0, 1)).toBe(0);
    expect(alphaAt(result, 5, 4, 1)).toBe(0);
  });

  it("shrinks the boundary of a full selection", () => {
    const selected = Array.from({ length: 25 }, (_, index) => [index % 5, Math.floor(index / 5)] as [number, number]);
    const result = shrinkMaskPixels(mask(5, 5, selected), 1);
    expect(alphaAt(result, 5, 0, 0)).toBe(0);
    expect(alphaAt(result, 5, 1, 1)).toBe(255);
    expect(alphaAt(result, 5, 3, 3)).toBe(255);
    expect(alphaAt(result, 5, 4, 4)).toBe(0);
  });

  it("feathers inward without extending the painted shape", () => {
    const result = featherMaskPixels(mask(5, 5, [[2, 2]]), 1);
    expect(alphaAt(result, 5, 2, 2)).toBeGreaterThan(0);
    expect(alphaAt(result, 5, 2, 2)).toBeLessThan(255);
    expect(alphaAt(result, 5, 1, 2)).toBe(0);
    expect(alphaAt(result, 5, 0, 0)).toBe(0);
  });
});
