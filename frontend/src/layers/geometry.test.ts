import { describe, expect, it } from "vitest";
import { Placement, canvasToLayer, containsPoint, generationPlacement, handlePoint, initialPlacement, layerToCanvas, moved, normalizeAngle, resized, rotated } from "./geometry";

const layer: Placement = { x: 100, y: 50, width: 40, height: 20, scale_x: 1, scale_y: 1, rotation: 0 };

function close(actual: { x: number; y: number }, expected: { x: number; y: number }) {
  expect(actual.x).toBeCloseTo(expected.x, 6);
  expect(actual.y).toBeCloseTo(expected.y, 6);
}

describe("layer geometry", () => {
  it("fits full generations to the canvas without distorting their aspect ratio", () => {
    expect(generationPlacement({ width: 1024, height: 1024 }, { width: 1024, height: 1024 })).toMatchObject({ x: 512, y: 512, scale_x: 1, scale_y: 1 });
    expect(generationPlacement({ width: 512, height: 256 }, { width: 1024, height: 1024 })).toMatchObject({ x: 512, y: 512, scale_x: 2, scale_y: 2 });
    expect(generationPlacement({ width: 2048, height: 1024 }, { width: 800, height: 600 })).toMatchObject({ x: 400, y: 300, scale_x: 800 / 2048, scale_y: 800 / 2048 });
  });
  it("maps layer pixels to the canvas and back through scale, flip and rotation", () => {
    const turned = { ...layer, scale_x: -2, scale_y: 0.5, rotation: 30 };
    const point = { x: 7, y: 3 };
    close(canvasToLayer(turned, layerToCanvas(turned, point)), point);
    close(layerToCanvas({ ...layer, rotation: 90 }, { x: 40, y: 10 }), { x: 100, y: 70 });
    expect(containsPoint(layer, { x: 119, y: 59 })).toBe(true);
    expect(containsPoint(layer, { x: 121, y: 50 })).toBe(false);
  });

  it("moves by the pointer travel", () => {
    expect(moved(layer, { x: 10, y: 10 }, { x: 15, y: 7 })).toMatchObject({ x: 105, y: 47 });
  });

  it("keeps the opposite corner fixed and the aspect ratio when dragging a corner", () => {
    const anchor = handlePoint(layer, "nw");
    const bigger = resized(layer, "se", { x: 160, y: 70 }, false);
    expect(bigger.scale_x).toBeCloseTo(bigger.scale_y, 6);
    expect(bigger.scale_x).toBeGreaterThan(1);
    close(handlePoint(bigger, "nw"), anchor);
  });

  it("stretches one side from an edge, and both sides with free", () => {
    const wider = resized(layer, "e", { x: 160, y: 50 }, false);
    expect(wider.scale_x).toBeCloseTo(2, 6);
    expect(wider.scale_y).toBe(1);
    close(handlePoint(wider, "w"), handlePoint(layer, "w"));
    const both = resized(layer, "e", { x: 160, y: 50 }, true);
    expect([both.scale_x, both.scale_y]).toEqual([2, 2]);
  });

  it("resizes a rotated and flipped layer along its own axes", () => {
    const turned = { ...layer, rotation: 90, scale_x: -1 };
    const target = handlePoint({ ...turned, scale_x: -1.5, x: turned.x, y: turned.y - 10 }, "e");
    const stretched = resized(turned, "e", target, false);
    expect(stretched.scale_x).toBeCloseTo(-1.5, 6);
    close(handlePoint(stretched, "w"), handlePoint(turned, "w"));
  });

  it("never collapses or flips a layer through zero", () => {
    const squashed = resized(layer, "e", { x: 50, y: 50 }, false);
    expect(squashed.scale_x).toBeCloseTo(0.01, 6);
  });

  it("rotates around the centre and snaps to 15 degrees", () => {
    expect(rotated(layer, { x: 150, y: 50 }, { x: 100, y: 100 }, false).rotation).toBeCloseTo(90, 6);
    expect(rotated(layer, { x: 150, y: 50 }, { x: 150, y: 61 }, true).rotation).toBe(15);
    expect(normalizeAngle(270)).toBe(-90);
  });

  it("places a new layer centred and fitted inside the canvas", () => {
    expect(initialPlacement({ width: 2000, height: 1000 }, { width: 1000, height: 800 })).toEqual({ x: 500, y: 400, scale_x: 0.4, scale_y: 0.4, rotation: 0 });
    expect(initialPlacement({ width: 100, height: 100 }, { width: 1000, height: 800 }).scale_x).toBe(1);
  });
});
