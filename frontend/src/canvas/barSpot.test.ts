import { describe, expect, it } from "vitest";
import { BAR_MARGIN, parseSpot, spotAt, spotStyle } from "./barSpot";

const bar = { width: 400, height: 34 }, frame = { width: 1000, height: 600 };

describe("where a moved bar stays", () => {
  it("measures the spot as a share of the free room", () => {
    const freeX = frame.width - bar.width - 2 * BAR_MARGIN, freeY = frame.height - bar.height - 2 * BAR_MARGIN;
    expect(spotAt(BAR_MARGIN, BAR_MARGIN, bar, frame)).toEqual({ x: 0, y: 0 });
    expect(spotAt(BAR_MARGIN + freeX / 2, BAR_MARGIN + freeY, bar, frame)).toEqual({ x: 0.5, y: 1 });
  });
  it("keeps the bar inside the frame when it is dragged past an edge", () => {
    expect(spotAt(-300, 5000, bar, frame)).toEqual({ x: 0, y: 1 });
  });
  it("pins a bar wider than the frame to the left, so Generate stays in view", () => {
    expect(spotAt(200, 100, { width: 1200, height: 34 }, frame).x).toBe(0);
  });
  it("places the bar by its own size, so it keeps its corner when the window resizes", () => {
    expect(spotStyle({ x: 1, y: 0.25 })).toEqual({
      left: `calc(${BAR_MARGIN}px + (100% - ${2 * BAR_MARGIN}px) * 1)`,
      top: `calc(${BAR_MARGIN}px + (100% - ${2 * BAR_MARGIN}px) * 0.25)`,
      transform: "translate(-100%, -25%)",
    });
  });
  it("reads a stored spot and ignores anything else", () => {
    expect(parseSpot(JSON.stringify({ x: 0.2, y: 1.4 }))).toEqual({ x: 0.2, y: 1 });
    expect(parseSpot(null)).toBeNull();
    expect(parseSpot("null")).toBeNull();
    expect(parseSpot("{\"x\": \"left\", \"y\": 0}")).toBeNull();
    expect(parseSpot("not json")).toBeNull();
  });
});
