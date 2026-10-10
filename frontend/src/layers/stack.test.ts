import { describe, expect, it } from "vitest";
import { LayerRecord } from "../api";
import {
  duplicateLayer, fillsCanvas, insertResult, layerAt, layerName, mergeBounds, mergePlan, moveLayer, patchLayer, rangeIds, removeLayers, replacementOf,
  splitVariants, stepVariant, variantIndex, withMerged,
} from "./stack";

function layer(id: string, values: Partial<LayerRecord> = {}): LayerRecord {
  return { id, name: id, image_path: `images/layers/layer-${id}.png`, mask_path: null, mask_enabled: true, variant_paths: [], variant_urls: [], x: 50, y: 50, scale_x: 1, scale_y: 1, rotation: 0, opacity: 1, visible: true, width: 20, height: 20, image_url: `/files/${id}.png`, mask_url: null, original_id: null, ...values };
}

const canvas = { width: 100, height: 100 };
const ids = (layers: LayerRecord[]) => layers.map(item => item.id);

describe("layer stack", () => {
  it("puts AI results on top, or above the layer they blend in and hides it", () => {
    const stack = [layer("a"), layer("b"), layer("c")];
    expect(insertResult(stack, layer("r"), null).map(item => item.id)).toEqual(["a", "b", "c", "r"]);
    const blended = insertResult(stack, layer("r"), "a");
    expect(blended.map(item => item.id)).toEqual(["a", "r", "b", "c"]);
    expect(blended.map(item => item.visible)).toEqual([false, true, true, true]);
    expect(blended[1].original_id).toBe("a");
    expect(replacementOf(blended, "a")?.id).toBe("r");
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
    expect(duplicateLayer([layer("a", { original_id: "x" })], "a")!.copy.original_id).toBeNull();
  });

  it("finds the topmost visible layer under a point", () => {
    const stack = [layer("low"), layer("high"), layer("hidden", { visible: false })];
    expect(layerAt(stack, { x: 50, y: 50 })?.id).toBe("high");
    expect(layerAt(stack, { x: 50, y: 50 }, item => item.id !== "high")?.id).toBe("low");
    expect(layerAt(stack, { x: 5, y: 5 })).toBeNull();
  });

  it("steps through a layer's variants and wraps around", () => {
    const paths = ["v1", "v2", "v3"], urls = ["/u1", "/u2", "/u3"];
    const shown = layer("g", { image_path: "v3", image_url: "/u3", variant_paths: paths, variant_urls: urls });
    expect(variantIndex(shown)).toBe(2);
    expect(stepVariant(shown, 1)).toEqual({ image_path: "v1", image_url: "/u1" });
    expect(stepVariant(shown, -1)).toEqual({ image_path: "v2", image_url: "/u2" });
    expect(stepVariant(layer("plain"), 1)).toBeNull();
    expect(variantIndex(layer("plain"))).toBe(-1);
  });

  it("splits variants into hidden layers just above, keeping the shown one in place", () => {
    const generative = layer("g", { image_path: "v2", image_url: "/u2", variant_paths: ["v1", "v2", "v3"], variant_urls: ["/u1", "/u2", "/u3"] });
    const result = splitVariants([layer("a"), generative, layer("b")], "g")!;
    expect(result.added).toBe(2);
    expect(result.layers.map(item => [item.name, item.image_path, item.visible, item.variant_paths.length])).toEqual([
      ["a", "images/layers/layer-a.png", true, 0], ["g", "v2", true, 0], ["g · 1", "v1", false, 0], ["g · 3", "v3", false, 0], ["b", "images/layers/layer-b.png", true, 0],
    ]);
    expect(splitVariants([layer("plain")], "plain")).toBeNull();
  });

  it("shows the original again when the AI layer that replaced it is deleted", () => {
    const stack = [layer("bg"), layer("a", { visible: false }), layer("r", { original_id: "a" }), layer("top")];
    const removed = removeLayers(stack, ["r"]);
    expect(ids(removed)).toEqual(["bg", "a", "top"]);
    expect(removed[1].visible).toBe(true);
    // Deleting the original instead leaves the AI layer on its own.
    expect(removeLayers(stack, ["a"])[1]).toMatchObject({ id: "r", original_id: null });
    expect(ids(removeLayers(stack, ["a", "r", "top"]))).toEqual(["bg"]);
  });

  it("selects the layers between two clicks in either direction", () => {
    const stack = [layer("a"), layer("b"), layer("c"), layer("d")];
    expect(rangeIds(stack, "b", "d")).toEqual(["b", "c", "d"]);
    expect(rangeIds(stack, "c", "a")).toEqual(["a", "b", "c"]);
    expect(rangeIds(stack, "gone", "c")).toEqual(["c"]);
  });

  it("knows a plain background that covers the canvas", () => {
    const background = layer("bg", { width: 100, height: 100 });
    expect(fillsCanvas(background, canvas)).toBe(true);
    expect(fillsCanvas({ ...background, x: 40 }, canvas)).toBe(false);
    expect(fillsCanvas({ ...background, opacity: 0.5 }, canvas)).toBe(false);
    expect(fillsCanvas({ ...background, mask_path: "m.png" }, canvas)).toBe(false);
    expect(fillsCanvas({ ...background, mask_path: "m.png", mask_enabled: false }, canvas)).toBe(true);
  });

  it("merges down onto the next visible layer, taking its place and name", () => {
    const stack = [layer("bg"), layer("hidden", { visible: false }), layer("a"), layer("faded", { opacity: 0 })];
    const plan = mergePlan(stack, "down", ["a"], canvas)!;
    expect([ids(plan.parts), plan.at, plan.name, plan.removed]).toEqual([["bg", "a"], "bg", "bg", ["bg", "a"]]);
    const merged = withMerged(stack, plan, layer("m"));
    expect(ids(merged)).toEqual(["m", "hidden", "faded"]);
    // A hidden layer, or one with nothing visible below, has nothing to merge.
    expect(mergePlan(stack, "down", ["hidden"], canvas)).toBeNull();
    expect(mergePlan(stack, "down", ["bg"], canvas)).toBeNull();
    expect(mergePlan(stack, "down", [], canvas)).toBeNull();
  });

  it("lets hidden originals go with the layers that replaced them", () => {
    const stack = [layer("bg"), layer("first", { visible: false }), layer("again", { visible: false, original_id: "first" }), layer("blend", { original_id: "again" }), layer("top")];
    const plan = mergePlan(stack, "down", ["blend"], canvas)!;
    expect(plan.removed).toEqual(["bg", "first", "again", "blend"]);
    expect(ids(withMerged(stack, plan, layer("m")))).toEqual(["m", "top"]);
    // An original that is shown again is merged like any other layer, and its own hidden original goes with it.
    const shown = mergePlan(patchLayer(stack, "again", { visible: true }), "down", ["blend"], canvas)!;
    expect([ids(shown.parts), shown.removed]).toEqual([["again", "blend"], ["first", "again", "blend"]]);
  });

  it("merges the visible selected layers at the top one's place", () => {
    const stack = [layer("bg"), layer("a"), layer("b", { visible: false }), layer("c"), layer("d")];
    const plan = mergePlan(stack, "selected", ["a", "b", "c"], canvas)!;
    expect([ids(plan.parts), plan.at, plan.name]).toEqual([["a", "c"], "c", "c"]);
    expect(ids(withMerged(stack, plan, layer("m")))).toEqual(["bg", "b", "m", "d"]);
    expect(mergePlan(stack, "selected", ["a", "b"], canvas)).toBeNull();
  });

  it("merges every visible layer at the bottom one's place and keeps hidden ones", () => {
    const stack = [layer("hidden", { visible: false }), layer("bg"), layer("a"), layer("b")];
    const plan = mergePlan(stack, "visible", [], canvas)!;
    expect([ids(plan.parts), plan.at, plan.name]).toEqual([["bg", "a", "b"], "bg", "bg"]);
    expect(ids(withMerged(stack, plan, layer("m")))).toEqual(["hidden", "m"]);
    expect(mergePlan([layer("bg"), layer("off", { visible: false })], "visible", [], canvas)).toBeNull();
  });

  it("flattens into one Background and discards hidden layers", () => {
    const stack = [layer("bg"), layer("hidden", { visible: false }), layer("a")];
    const plan = mergePlan(stack, "flatten", [], canvas)!;
    expect([ids(plan.parts), plan.name, plan.removed.length]).toEqual([["bg", "a"], "Background", 3]);
    expect(ids(withMerged(stack, plan, layer("m")))).toEqual(["m"]);
    // A lone layer that already is the canvas has nothing to flatten; a moved one does.
    expect(mergePlan([layer("bg", { width: 100, height: 100 })], "flatten", [], canvas)).toBeNull();
    expect(mergePlan([layer("bg")], "flatten", [], canvas)).not.toBeNull();
  });

  it("keeps what merged layers cover, up to half a canvas past its edges", () => {
    expect(mergeBounds([layer("a"), layer("b", { x: 90, y: 20 })], canvas, false)).toEqual({ x: 40, y: 10, width: 60, height: 50 });
    expect(mergeBounds([layer("turned", { rotation: 45 })], canvas, false)).toEqual({ x: 35, y: 35, width: 30, height: 30 });
    expect(mergeBounds([layer("huge", { scale_x: 20, scale_y: 20 })], canvas, false)).toEqual({ x: -50, y: -50, width: 200, height: 200 });
    expect(mergeBounds([layer("a")], canvas, true)).toEqual({ x: 0, y: 0, width: 100, height: 100 });
  });
});
