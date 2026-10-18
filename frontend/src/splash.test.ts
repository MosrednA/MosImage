import { describe, expect, it } from "vitest";
import { HOLD_MS, glide, holdLeft } from "./splash";

describe("splash", () => {
  it("moves the centred logo onto the start screen's, whatever their sizes", () => {
    const centred = { left: 608, top: 388, width: 64, height: 64 };
    expect(glide(centred, { left: 608, top: 300, width: 64, height: 64 })).toEqual({ x: 0, y: -88 });
    expect(glide(centred, { left: 600, top: 380, width: 80, height: 80 })).toEqual({ x: 0, y: 0 });
  });
  it("sits centred a moment on a fast load and not at all on a slow one", () => {
    expect(holdLeft(120)).toBe(HOLD_MS - 120);
    expect(holdLeft(HOLD_MS + 500)).toBe(0);
  });
});
