import { describe, expect, it } from "vitest";
import { sketchFootprint, sketchReach } from "./sketch";

describe("sketch footprint", () => {
  it("reaches about 2.5% of the picture past the strokes", () => {
    expect(sketchReach({ width: 1600, height: 900 })).toBe(40);
    expect(sketchReach({ width: 200, height: 100 })).toBe(8);
    expect(sketchReach({ width: 8000, height: 6000 })).toBe(64);
  });

  it("covers the strokes fully and fades out beyond the reach", async () => {
    const size = { width: 60, height: 40 };
    const rgba = new Uint8ClampedArray(size.width * size.height * 4);
    for (let x = 28; x < 32; x += 1) rgba[(20 * size.width + x) * 4 + 3] = 255;
    const footprint = (await sketchFootprint(rgba, size, 5))!;
    const at = (x: number, y: number) => footprint.alpha[(y - footprint.box.y) * footprint.box.width + (x - footprint.box.x)];
    expect(at(30, 20)).toBe(255);
    expect(at(30, 24)).toBe(255);
    expect(at(30, 27)).toBeGreaterThan(0);
    expect(at(30, 27)).toBeLessThan(255);
    expect(at(30, 32)).toBe(0);
    expect(await sketchFootprint(new Uint8ClampedArray(size.width * size.height * 4), size, 5)).toBeNull();
  });
});
