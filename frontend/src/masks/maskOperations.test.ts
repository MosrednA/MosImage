import { describe, expect, it } from "vitest";
import { invertMaskPixels } from "./maskOperations";

describe("mask pixel operations", () => {
  it("inverts alpha while keeping a white mask", () => {
    const data = new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 0]);
    const result = invertMaskPixels({ width: 2, height: 1, data });
    expect(Array.from(result)).toEqual([255, 255, 255, 0, 255, 255, 255, 255]);
  });

  it("rejects a buffer that does not match its dimensions", () => {
    expect(() => invertMaskPixels({ width: 3, height: 1, data: new Uint8ClampedArray(8) })).toThrow();
  });
});
