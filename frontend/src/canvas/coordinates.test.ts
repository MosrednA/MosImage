import { describe, expect, it } from "vitest";
import { viewportPointToImage } from "./coordinates";

describe("viewportPointToImage", () => {
  it("maps a scaled landscape viewport to native image coordinates", () => {
    const point = viewportPointToImage(
      { x: 500, y: 250 },
      { left: 100, top: 50, width: 800, height: 400 },
      { width: 1600, height: 800 },
    );
    expect(point).toEqual({ x: 800, y: 400 });
  });

  it("clamps letterbox-adjacent pointer coordinates to image bounds", () => {
    const point = viewportPointToImage(
      { x: 10, y: 900 },
      { left: 100, top: 50, width: 300, height: 600 },
      { width: 600, height: 1200 },
    );
    expect(point).toEqual({ x: 0, y: 1200 });
  });
});
