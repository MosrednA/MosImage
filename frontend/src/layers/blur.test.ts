import { describe, expect, it } from "vitest";
import { blurChannel, boxSizes } from "./blur";

describe("script blur", () => {
  it("picks three box widths whose spread matches the Gaussian", () => {
    for (const sigma of [1, 2.5, 8, 24]) {
      const sizes = boxSizes(sigma);
      expect(sizes).toHaveLength(3);
      expect(sizes.every(size => size % 2 === 1)).toBe(true);
      // Each box of width w adds (w² - 1) / 12 to the variance.
      const variance = sizes.reduce((total, size) => total + (size * size - 1) / 12, 0);
      expect(Math.sqrt(variance)).toBeCloseTo(sigma, 0);
    }
  });

  it("spreads a solid square into a soft edge without moving its centre", () => {
    const width = 40, height = 30;
    const channel = new Uint8ClampedArray(width * height);
    for (let y = 10; y < 20; y += 1) for (let x = 15; x < 25; x += 1) channel[y * width + x] = 255;
    const blurred = blurChannel(channel, width, height, 2);
    expect(blurred[15 * width + 20]).toBeGreaterThan(200);
    const edge = blurred[15 * width + 14];
    expect(edge).toBeGreaterThan(0);
    expect(edge).toBeLessThan(255);
    expect(blurred[15 * width + 4]).toBe(0);
    // Symmetric: the left and right edges fade alike.
    expect(blurred[15 * width + 12]).toBe(blurred[15 * width + 27]);
  });

  it("treats pixels outside as empty and leaves a zero radius alone", () => {
    const full = new Uint8ClampedArray(11 * 11).fill(255);
    const blurred = blurChannel(full, 11, 11, 2);
    expect(blurred[5 * 11 + 5]).toBe(255);
    expect(blurred[0]).toBeLessThan(blurred[5 * 11 + 5] / 2);
    expect(Array.from(blurChannel(full, 11, 11, 0))).toEqual(Array.from(full));
  });
});
