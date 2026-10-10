import { describe, expect, it } from "vitest";
import { LayerRecord } from "../api";
import { duplicateLayer, insertResult, layerAt, layerName, moveLayer, patchLayer } from "./stack";

function layer(id: string, values: Partial<LayerRecord> = {}): LayerRecord {
  return { id, name: id, image_path: `images/layers/layer-${id}.png`, mask_path: null, mask_enabled: true, x: 50, y: 50, scale_x: 1, scale_y: 1, rotation: 0, opacity: 1, visible: true, width: 20, height: 20, image_url: `/files/${id}.png`, mask_url: null, ...values };
}

describe("layer stack", () => {
  it("puts AI results on top, or above the layer they blend in and hides it", () => {
    const stack = [layer("a"), layer("b"), layer("c")];
    expect(insertResult(stack, layer("r"), null).map(item => item.id)).toEqual(["a", "b", "c", "r"]);
    const blended = insertResult(stack, layer("r"), "a");
    expect(blended.map(item => item.id)).toEqual(["a", "r", "b", "c"]);
    expect(blended.map(item => item.visible)).toEqual([false, true, true, true]);
    expect(insertResult(stack, layer("r"), "gone").map(item => item.id)).toEqual(["a", "b", "c", "r"]);
  });

  it("names layers after their file", () => {
    expect(layerName("anna_on-beach.png", 3)).toBe("anna on beach");
    expect(layerName("image.png", 3)).toBe("Layer 3");
    expect(layerName(undefined, 1)).toBe("Layer 1");
  });

  it("reorders and patches layers", () => {
    const stack = [layer("a"), layer("b"), layer("c")];
    expect(moveLayer(stack, "a", 2).map(item => item.id)).toEqual(["b", "c", "a"]);
    expect(moveLayer(stack, "c", 0).map(item => item.id)).toEqual(["c", "a", "b"]);
    expect(patchLayer(stack, "b", { opacity: 0.5 })[1].opacity).toBe(0.5);
  });

  it("duplicates just above the original with a new id", () => {
    const result = duplicateLayer([layer("a"), layer("b")], "a")!;
    expect(result.layers.map(item => item.name)).toEqual(["a", "a copy", "b"]);
    expect(result.copy.id).not.toBe("a");
    expect([result.copy.x, result.copy.image_path]).toEqual([74, "images/layers/layer-a.png"]);
  });

  it("finds the topmost visible layer under a point", () => {
    const stack = [layer("low"), layer("high"), layer("hidden", { visible: false })];
    expect(layerAt(stack, { x: 50, y: 50 })?.id).toBe("high");
    expect(layerAt(stack, { x: 50, y: 50 }, item => item.id !== "high")?.id).toBe("low");
    expect(layerAt(stack, { x: 5, y: 5 })).toBeNull();
  });
});
