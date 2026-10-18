import { describe, expect, it } from "vitest";
import { CommandState, canSubmit, commandHint } from "./commandState";

const base: CommandState = {
  operation: "replace", prompt: "a hat", hasImage: true, hasMask: true, hasSketch: false, expansionValid: false,
  editScope: "visible", editLayerName: null, sketchScope: "strokes", referenceNames: [], tool: "object",
  activeLayerName: null, layerMasking: false, blendingName: null,
};
const state = (overrides: Partial<CommandState>): CommandState => ({ ...base, ...overrides });

describe("canSubmit", () => {
  it("needs a prompt and a selection for regional edits, except removal", () => {
    expect(canSubmit(base)).toBe(true);
    expect(canSubmit(state({ hasMask: false }))).toBe(false);
    expect(canSubmit(state({ prompt: " " }))).toBe(false);
    expect(canSubmit(state({ operation: "remove", prompt: "" }))).toBe(true);
  });
  it("needs strokes for a sketch and only a picture for faces", () => {
    expect(canSubmit(state({ referenceNames: ["Anna"] }))).toBe(true);
    expect(canSubmit(state({ operation: "sketch", hasSketch: true }))).toBe(true);
    expect(canSubmit(state({ operation: "sketch", hasSketch: false }))).toBe(false);
    expect(canSubmit(state({ operation: "faces", prompt: "", hasMask: false }))).toBe(true);
    expect(canSubmit(state({ operation: "faces", hasImage: false }))).toBe(false);
    expect(canSubmit(state({ operation: "faces", facePicks: { found: 3, picked: 0 } }))).toBe(false);
    expect(canSubmit(state({ operation: "faces", facePicks: { found: 3, picked: 1 } }))).toBe(true);
    expect(canSubmit(state({ operation: "enhance", prompt: "", hasMask: false }))).toBe(true);
    expect(canSubmit(state({ operation: "enhance", hasImage: false }))).toBe(false);
    expect(commandHint(state({ operation: "faces", faceMode: "redraw" }))).toContain("drawn anew");
    expect(commandHint(state({ operation: "enhance", hasMask: false }))).toContain("the whole picture");
  });
  it("edits the visible picture, or a picked layer", () => {
    expect(canSubmit(state({ operation: "edit", hasMask: false }))).toBe(true);
    expect(canSubmit(state({ operation: "edit", hasMask: false, editScope: "layer" }))).toBe(false);
    expect(canSubmit(state({ operation: "edit", hasMask: false, editScope: "layer", editLayerName: "Hat" }))).toBe(true);
  });
  it("expands to a valid size with a prompt", () => {
    expect(canSubmit(state({ operation: "expand", expansionValid: true }))).toBe(true);
    expect(canSubmit(state({ operation: "expand" }))).toBe(false);
  });
  it("generates from a prompt alone", () => {
    expect(canSubmit(state({ operation: "generate", hasImage: false, hasMask: false }))).toBe(true);
  });
});

describe("commandHint", () => {
  it("puts the layer being worked on first", () => {
    expect(commandHint(state({ activeLayerName: "Hat", layerMasking: true }))).toContain("Painting the mask of “Hat”");
    expect(commandHint(state({ activeLayerName: "Hat" }))).toContain("Editing layer “Hat”");
    // The Brush selects on the picture even with a layer selected; masks are painted from the Layers panel.
    expect(commandHint(state({ activeLayerName: "Hat" }))).toContain("Brush or another selection tool selects on the picture");
  });
  it("explains blending and that results become layers", () => {
    expect(commandHint(state({ blendingName: "Hat" }))).toContain("Blending “Hat” in");
    expect(commandHint(base)).toContain("Apply adds the result as a new layer.");
    // Whole-picture hints say it themselves, once.
    expect(commandHint(state({ operation: "edit" })).split("Apply adds the result as a new layer.")).toHaveLength(2);
  });
});
