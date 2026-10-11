import { describe, expect, it } from "vitest";
import { footprintPixels, luminanceToAlpha, thresholdAlpha } from "./bitmaps";

describe("layer bitmaps", () => {
  it("stores greyscale mask coverage in alpha over white", () => {
    const data = new Uint8ClampedArray([0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255]);
    luminanceToAlpha(data);
    expect(Array.from(data)).toEqual([255, 255, 255, 0, 255, 255, 255, 128, 255, 255, 255, 255]);
  });

  it("grows a soft footprint into a solid one", () => {
    const data = new Uint8ClampedArray([0, 0, 0, 0, 10, 20, 30, 8, 255, 255, 255, 9, 0, 0, 0, 200]);
    thresholdAlpha(data, 8);
    expect(Array.from(data)).toEqual([255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255]);
  });

  it("grows and softens a footprint without canvas filters", () => {
    const width = 30, height = 30;
    const data = new Uint8ClampedArray(width * height * 4);
    for (let y = 10; y < 20; y += 1) for (let x = 10; x < 20; x += 1) data[(y * width + x) * 4 + 3] = 255;
    footprintPixels(data, width, height, 8, 2);
    const grey = (x: number, y: number) => data[(y * width + x) * 4];
    expect(grey(15, 15)).toBe(255);
    // Grown by about the blur radius, then softened at the new edge.
    expect(grey(8, 15)).toBeGreaterThan(128);
    expect(grey(5, 15)).toBeGreaterThan(0);
    expect(grey(5, 15)).toBeLessThan(255);
    expect(grey(0, 0)).toBe(0);
    expect(data[3]).toBe(255);
  });
});
