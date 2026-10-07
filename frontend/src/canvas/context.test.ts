import { describe, expect, it } from "vitest";
import { maskBounds, paddedContextBox } from "./context";

describe("editable mask and protected context", () => {
  it("uses alpha and includes the last pixel on a non-square mask", () => {
    const data = new Uint8ClampedArray(6 * 4 * 4);
    data.fill(255);
    for (let pixel = 0; pixel < 24; pixel += 1) data[pixel * 4 + 3] = 0;
    data[(1 * 6 + 2) * 4 + 3] = 255;
    data[(3 * 6 + 5) * 4 + 3] = 1;
    expect(maskBounds({ width: 6, height: 4, data })).toEqual({ x: 2, y: 1, width: 4, height: 3 });
  });
  it("has no context rectangle for an empty selection", () => {
    expect(maskBounds({ width: 2, height: 3, data: new Uint8ClampedArray(24) })).toBeNull();
  });
  it("clips only the context rectangle at the image boundary", () => {
    expect(paddedContextBox({ x: 136, y: 924, width: 236, height: 262 }, { width: 1254, height: 1254 }, 192))
      .toEqual({ x: 0, y: 732, width: 564, height: 522 });
  });
});
