import { describe, expect, it } from "vitest";
import { expansionForRatio, expansionSize, validExpansion } from "./expansion";

describe("native-resolution expansion", () => {
  it("expands without cropping and distributes odd pixels exactly", () => {
    const source = { width: 101, height: 100 };
    const margins = expansionForRatio(source, 16 / 9);
    expect(margins).toEqual({ left: 38, right: 39, top: 0, bottom: 0 });
    expect(expansionSize(source, margins)).toEqual({ width: 178, height: 100 });
  });
  it("rejects empty, fractional, shrinking and oversized canvases", () => {
    const source = { width: 3000, height: 2000 };
    expect(validExpansion(source, { left: 0, right: 0, top: 0, bottom: 0 })).toBe(false);
    expect(validExpansion(source, { left: -.5, right: 1, top: 0, bottom: 0 })).toBe(false);
    expect(validExpansion(source, { left: 0, right: 1200, top: 0, bottom: 0 })).toBe(false);
    expect(validExpansion(source, { left: 128, right: 128, top: 0, bottom: 0 })).toBe(true);
  });
});
