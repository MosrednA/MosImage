import { describe, expect, it } from "vitest";
import { luminanceToAlpha } from "./bitmaps";

describe("layer bitmaps", () => {
  it("stores greyscale mask coverage in alpha over white", () => {
    const data = new Uint8ClampedArray([0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255]);
    luminanceToAlpha(data);
    expect(Array.from(data)).toEqual([255, 255, 255, 0, 255, 255, 255, 128, 255, 255, 255, 255]);
  });
});
