import { describe, expect, it } from "vitest";
import { addsSpace, cropBounds, cropForRatio, dragCrop, fullCrop, ratioValue, resizeCrop, validCrop } from "./crop";

const size = { width: 1000, height: 600 };

describe("canvas crop", () => {
  it("drags edges and corners while the opposite side stays put", () => {
    const start = fullCrop(size);
    expect(dragCrop(start, "w", 100, 0, null)).toEqual({ x: 100, y: 0, width: 900, height: 600 });
    expect(dragCrop(start, "se", -200, -100, null)).toEqual({ x: 0, y: 0, width: 800, height: 500 });
    expect(dragCrop(start, "move", -30.4, 12.6, null)).toEqual({ x: -30, y: 13, width: 1000, height: 600 });
  });

  it("grows past the picture, which adds transparent space", () => {
    const grown = dragCrop(fullCrop(size), "n", 0, -150, null);
    expect(grown).toEqual({ x: 0, y: -150, width: 1000, height: 750 });
    expect(addsSpace(size, grown)).toBe(true);
    expect(validCrop(size, grown)).toBe(true);
  });

  it("keeps a chosen ratio from edges and corners", () => {
    const square = cropForRatio(size, 1);
    expect(square).toEqual({ x: 200, y: 0, width: 600, height: 600 });
    const edge = dragCrop(square, "e", -100, 0, 1);
    expect(edge).toEqual({ x: 200, y: 50, width: 500, height: 500 });
    const corner = dragCrop(square, "nw", 50, 200, 1);
    expect(corner).toEqual({ x: 400, y: 200, width: 400, height: 400 });
    expect(ratioValue("16:9", size)).toBeCloseTo(16 / 9);
    expect(ratioValue("original", size)).toBeCloseTo(1000 / 600);
    expect(ratioValue("free", size)).toBeNull();
  });

  it("never shrinks below 16 px or grows past 4096 px", () => {
    expect(dragCrop(fullCrop(size), "e", -5000, 0, null).width).toBe(16);
    expect(dragCrop(fullCrop(size), "s", 0, 9000, null).height).toBe(4096);
    expect(resizeCrop(fullCrop(size), 3000, 600, 1, "width")).toEqual({ x: -1000, y: -1200, width: 3000, height: 3000 });
  });

  it("accepts only crops that change the canvas and keep part of the picture", () => {
    expect(validCrop(size, fullCrop(size))).toBe(false);
    expect(validCrop(size, { x: 1000, y: 0, width: 100, height: 100 })).toBe(false);
    expect(validCrop(size, { x: 10.5, y: 0, width: 100, height: 100 })).toBe(false);
    expect(validCrop(size, { x: 10, y: 0, width: 100, height: 100 })).toBe(true);
  });

  it("shows room around the picture and the whole crop", () => {
    expect(cropBounds(size, fullCrop(size))).toEqual({ x: -120, y: -120, width: 1240, height: 840 });
    const wide = cropBounds(size, { x: -500, y: 0, width: 1500, height: 600 });
    expect(wide.x).toBe(-540);
  });
});
