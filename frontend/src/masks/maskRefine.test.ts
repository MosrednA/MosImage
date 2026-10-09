import { describe, expect, it } from "vitest";
import { alphaBounds, autoSoften, cropSelection, distanceField, edgeAlpha, refineAlpha, signedDistanceField } from "./maskRefine";
import { signedDistanceFieldAsync } from "./fieldWorker";

/** A width × height RGBA mask with a selected axis-aligned square. */
function square(width: number, height: number, left: number, top: number, size: number) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = top; y < top + size; y += 1) for (let x = left; x < left + size; x += 1) data[(y * width + x) * 4 + 3] = 255;
  return data;
}

function alphaAt(alpha: Uint8ClampedArray, field: { box: { x: number; y: number; width: number } }, x: number, y: number) {
  return alpha[(y - field.box.y) * field.box.width + (x - field.box.x)];
}

describe("non-destructive mask edges", () => {
  const data = square(40, 30, 15, 10, 10);
  const bounds = { x: 15, y: 10, width: 10, height: 10 };

  it("keeps the painted selection unchanged without expand or soften", () => {
    const field = signedDistanceField(data, 40, 30, bounds, 8);
    const alpha = refineAlpha(field, { expand: 0, soften: 0 });
    expect(alphaBounds(alpha, field.box)).toEqual(bounds);
    expect(alphaAt(alpha, field, 15, 15)).toBe(255);
    expect(alphaAt(alpha, field, 14, 15)).toBe(0);
  });

  it("expands and contracts by whole pixels", () => {
    const field = signedDistanceField(data, 40, 30, bounds, 8);
    expect(alphaBounds(refineAlpha(field, { expand: 3, soften: 0 }), field.box)).toEqual({ x: 12, y: 7, width: 16, height: 16 });
    expect(alphaBounds(refineAlpha(field, { expand: -2, soften: 0 }), field.box)).toEqual({ x: 17, y: 12, width: 6, height: 6 });
    expect(alphaBounds(refineAlpha(field, { expand: -6, soften: 0 }), field.box)).toBeNull();
  });

  it("softens outward so the whole selection is regenerated", () => {
    const field = signedDistanceField(data, 40, 30, bounds, 8);
    const alpha = refineAlpha(field, { expand: 0, soften: 4 });
    expect(alphaAt(alpha, field, 15, 15)).toBe(255);
    const ramp = [14, 13, 12, 11].map(x => alphaAt(alpha, field, x, 15));
    expect(ramp[0]).toBeLessThan(255);
    expect(ramp[0]).toBeGreaterThan(ramp[1]);
    expect(ramp[2]).toBeGreaterThan(ramp[3]);
    expect(alphaAt(alpha, field, 10, 15)).toBe(0);
  });

  it("clips the field at the image edge", () => {
    const edge = square(20, 20, 0, 0, 5);
    const field = signedDistanceField(edge, 20, 20, { x: 0, y: 0, width: 5, height: 5 }, 100);
    expect(field.box).toEqual({ x: 0, y: 0, width: 20, height: 20 });
  });

  it("uses a smooth ramp from transparent to opaque", () => {
    expect(edgeAlpha(1, 10)).toBe(255);
    expect(edgeAlpha(-10, 10)).toBe(0);
    expect(edgeAlpha(-5, 10)).toBe(128);
  });

  it("scales the automatic soften with the image", () => {
    expect(autoSoften(1024, 768)).toBe(26);
    expect(autoSoften(200, 200)).toBe(8);
    expect(autoSoften(8000, 4000)).toBe(64);
  });
});

describe("distance field halves", () => {
  it("match the one-step field, also through the worker client", async () => {
    const width = 40, height = 30;
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let y = 10; y < 18; y += 1) for (let x = 12; x < 25; x += 1) rgba[(y * width + x) * 4 + 3] = 255;
    const bounds = { x: 12, y: 10, width: 13, height: 8 };
    const whole = signedDistanceField(rgba, width, height, bounds, 6);
    const crop = cropSelection(rgba, width, height, bounds, 6);
    expect(distanceField(crop.box, crop.selected)).toEqual(whole);
    expect(await signedDistanceFieldAsync(rgba, width, height, bounds, 6)).toEqual(whole);
  });
});
