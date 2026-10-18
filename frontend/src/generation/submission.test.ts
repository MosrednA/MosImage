import { beforeEach, describe, expect, it, vi } from "vitest";
import { editWholeImage, enhanceImage, fixFaces, generateImage, inpaintImage, outpaintImage } from "../api";
import { defaultSettings, qualityOptions } from "../components/EditorSettings";
import { MAX_LAYERS } from "../layers/stack";
import { BLEND_STRENGTH, SubmissionInput, requestPrompt, resultName, submitJob } from "./submission";

vi.mock("../api", async original => {
  const created = { id: "job", status: "queued" };
  return {
    ...await original<typeof import("../api")>(),
    editWholeImage: vi.fn(async () => created), enhanceImage: vi.fn(async () => created), fixFaces: vi.fn(async () => created), generateImage: vi.fn(async () => created),
    inpaintImage: vi.fn(async () => created), outpaintImage: vi.fn(async () => created),
  };
});
vi.mock("../canvas/sketch", () => ({
  sketchMask: vi.fn(async (sketch: Blob) => sketch.size ? new Blob(["footprint"]) : null),
  sketchedPicture: vi.fn(async () => new File(["sketched"], "sketched.png")),
}));

const picture = new File(["picture"], "picture.png");
const layer = { id: "l1", name: "Hat", visible: true, opacity: 1 } as SubmissionInput["blendLayer"];

function input(overrides: Partial<SubmissionInput>): SubmissionInput {
  return {
    operation: "replace", settings: { ...defaultSettings, prompt: "a red hat" }, referenceIds: [], source: picture, size: { width: 100, height: 80 },
    hasDocument: true, layerCount: 0, mask: new Blob(["mask"]), sketch: null, sketchScope: "strokes", editScope: "visible", faceMode: "faithful", faces: null, enhanceLevel: "subtle", editLayer: null, blendLayer: null,
    expansion: { left: 128, top: 0, right: 128, bottom: 0 }, picture: async () => picture, ...overrides,
  };
}
function sentForm(): FormData { return vi.mocked(inpaintImage).mock.calls.at(-1)![0]; }

beforeEach(() => vi.clearAllMocks());

describe("prompts and names", () => {
  it("fills in a removal prompt", () => {
    expect(requestPrompt("remove", { ...defaultSettings, prompt: " " })).toBe("Continue the surrounding background.");
    expect(requestPrompt("remove", { ...defaultSettings, prompt: " x " })).toBe("x");
    expect(requestPrompt("replace", { ...defaultSettings, prompt: "  hat " })).toBe("hat");
  });
  it("names results after the prompt or the blended layer", () => {
    expect(resultName(null, "replace", "a big red party hat with stars")).toBe("AI · a big red party hat");
    expect(resultName("Hat", "replace", "x")).toBe("Hat blended");
    expect(resultName(null, "remove", "x")).toBe("AI removal");
    expect(resultName(null, "edit", "")).toBe("AI edit");
  });
});

describe("submitJob", () => {
  it("inpaints a regional edit with the final soft mask and its references", async () => {
    const submission = await submitJob(input({ referenceIds: [3, 4] }));
    const form = sentForm();
    expect(form.get("edit_operation")).toBe("replace");
    expect(form.get("blend_feather")).toBe("0");
    expect(form.get("strength")).toBe("1");
    expect(form.getAll("reference_ids")).toEqual(["3", "4"]);
    expect(submission.extras.result).toMatchObject({ kind: "selection", name: "AI · a red hat", replaces: null });
  });
  it("caps the strength when blending a layer in", async () => {
    const submission = await submitJob(input({ blendLayer: layer }));
    expect(sentForm().get("strength")).toBe(String(BLEND_STRENGTH));
    expect(submission.extras.result).toMatchObject({ name: "Hat blended", replaces: "l1" });
  });
  it("redraws only the sketch at full strength, or the whole picture through Edit", async () => {
    const sketch = new Blob(["strokes"]);
    const strokes = await submitJob(input({ operation: "sketch", sketch }));
    expect(sentForm().get("edit_operation")).toBe("sketch");
    expect(sentForm().get("strength")).toBe("1");
    expect(sentForm().get("scene_context")).toBeNull();
    expect(strokes.extras.result?.kind).toBe("selection");
    const whole = await submitJob(input({ operation: "sketch", sketch, sketchScope: "whole" }));
    expect(vi.mocked(editWholeImage).mock.calls[0][1]).toMatchObject({ mode: "sketch" });
    expect(whole.extras.result?.kind).toBe("generation");
    await expect(submitJob(input({ operation: "sketch", sketch: new Blob([]) }))).rejects.toThrow("Sketch something first");
  });
  it("sends the picked references with a sketch in either scope", async () => {
    const sketch = new Blob(["strokes"]);
    await submitJob(input({ operation: "sketch", sketch, referenceIds: [5, 6] }));
    expect(sentForm().getAll("reference_ids")).toEqual(["5", "6"]);
    await submitJob(input({ operation: "sketch", sketch, sketchScope: "whole", referenceIds: [5] }));
    expect(vi.mocked(editWholeImage).mock.calls[0][1]).toMatchObject({ mode: "sketch", reference_ids: [5] });
  });
  it("limits fixed faces to the selection", async () => {
    const mask = new Blob(["faces"]);
    const submission = await submitJob(input({ operation: "faces", mask, referenceIds: [7] }));
    expect(vi.mocked(fixFaces).mock.calls[0][1].reference_ids).toEqual([7]);
    expect(vi.mocked(fixFaces).mock.calls[0][2]).toBe(mask);
    expect(submission.extras.result).toEqual({ kind: "faces", name: "Fixed faces" });
    expect(vi.mocked(fixFaces).mock.calls[0][1].faithful).toBe(true);
    await submitJob(input({ operation: "faces", faceMode: "redraw" }));
    expect(vi.mocked(fixFaces).mock.calls[1][1].faithful).toBe(false);
    expect(vi.mocked(fixFaces).mock.calls[1][1].faces).toBeUndefined();
  });
  it("sends the faces picked on the canvas instead of the selection", async () => {
    await submitJob(input({ operation: "faces", mask: new Blob(["faces"]), faces: [[10, 20, 40, 60]] }));
    expect(vi.mocked(fixFaces).mock.calls[0][1].faces).toEqual([[10, 20, 40, 60]]);
    expect(vi.mocked(fixFaces).mock.calls[0][2]).toBeNull();
    await expect(submitJob(input({ operation: "faces", faces: [] }))).rejects.toThrow("Click a face");
  });
  it("enhances the selection or the whole picture on the chosen model, without references", async () => {
    const mask = new Blob(["detail"]);
    const selected = await submitJob(input({ operation: "enhance", mask, referenceIds: [7], settings: { ...defaultSettings, model_name: "flux2-klein-4b-nvfp4", prompt: "" } }));
    const [, parameters, selection] = vi.mocked(enhanceImage).mock.calls[0];
    expect(parameters).toMatchObject({ model_name: "flux2-klein-4b-nvfp4", detail: 0.4, prompt: "" });
    expect(parameters).not.toHaveProperty("reference_ids");
    expect(selection).toBe(mask);
    expect(selected.extras.result).toEqual({ kind: "selection", mask, name: "Enhanced", replaces: null });
    const whole = await submitJob(input({ operation: "enhance", mask: null, enhanceLevel: "strong" }));
    expect(vi.mocked(enhanceImage).mock.calls[1][1].detail).toBe(0.7);
    expect(vi.mocked(enhanceImage).mock.calls[1][2]).toBeNull();
    expect(whole.extras.result).toEqual({ kind: "generation", name: "Enhanced" });
  });
  it("keeps the chosen model for Enhance and Fix faces", () => {
    const fast = { ...defaultSettings, model_name: "flux2-klein-4b-nvfp4", consistency_lora: true };
    expect(qualityOptions(fast, "enhance", false)).toMatchObject({ model_name: "flux2-klein-4b-nvfp4", consistency_lora: false });
    expect(qualityOptions(fast, "faces", false).model_name).toBe("flux2-klein-4b-nvfp4");
    expect(qualityOptions(defaultSettings, "enhance", false).model_name).toBe("flux2-klein-9b-nvfp4");
  });
  it("opens a new image from the start screen and adds a layer in a document", async () => {
    const start = await submitJob(input({ operation: "generate", hasDocument: false, source: null, referenceIds: [5] }));
    expect(start.extras.result).toBeUndefined();
    const inDocument = await submitJob(input({ operation: "generate", referenceIds: [5] }));
    expect(inDocument.extras.result?.kind).toBe("generation");
    expect(vi.mocked(generateImage).mock.calls[0][0]).toMatchObject({ width: 1024, height: 1024, reference_ids: [5] });
  });
  it("needs a layer for a layer edit and enables diffMask for whole-picture edits", async () => {
    await expect(submitJob(input({ operation: "edit", editScope: "layer" }))).rejects.toThrow("Pick the layer");
    const whole = await submitJob(input({ operation: "edit" }));
    expect(whole.extras.result).toEqual({ kind: "generation", name: "AI · a red hat", diffMask: true });
  });
  it("expands the picture as you see it and moves layers along", async () => {
    const visible = new File(["with layers"], "picture.png");
    const submission = await submitJob(input({ operation: "expand", expansion: { left: 64, top: 32, right: 0, bottom: 0 }, picture: async () => visible }));
    expect(vi.mocked(outpaintImage).mock.calls[0][0]).toBe(visible);
    expect(submission).toMatchObject({ operation: "outpaint", extras: { layerOffset: { x: 64, y: 32 } } });
  });
  it("refuses another layer at the limit", async () => {
    await expect(submitJob(input({ layerCount: MAX_LAYERS }))).rejects.toThrow(`up to ${MAX_LAYERS} layers`);
  });
  it("uses Fast four-step outpaint and restores Quality when switched off", async () => {
    const settings = { ...defaultSettings, prompt: "continue", consistency_lora: true, specialized_outpaint: true, steps: 8, guidance: 2 };
    await submitJob(input({ operation: "expand", settings }));
    expect(vi.mocked(outpaintImage).mock.calls[0][1]).toMatchObject({ model_name: "flux2-klein-4b-nvfp4", steps: 4, guidance: 1, specialized_outpaint: true, consistency_lora: false });
    await submitJob(input({ operation: "expand", settings: { ...settings, specialized_outpaint: false } }));
    expect(vi.mocked(outpaintImage).mock.calls[1][1]).toMatchObject({ model_name: "flux2-klein-9b-nvfp4", steps: 8, guidance: 2, specialized_outpaint: false, consistency_lora: true });
  });
  it("sends crop detail at full strength and keeps Blend in capped", async () => {
    const settings = { ...defaultSettings, prompt: "weave", detail_crop: true, consistency_lora: true, strength: .5, context_padding: 96 };
    await submitJob(input({ settings }));
    expect(sentForm().get("detail_crop")).toBe("true");
    expect(sentForm().get("consistency_lora")).toBe("true");
    expect(sentForm().get("strength")).toBe("1");
    expect(sentForm().get("context_padding")).toBe("96");
    await submitJob(input({ settings, blendLayer: layer }));
    expect(sentForm().get("detail_crop")).toBe("false");
    expect(sentForm().get("strength")).toBe("0.5");
    await submitJob(input({ settings, operation: "remove" }));
    expect(sentForm().get("detail_crop")).toBe("false");
    expect(sentForm().get("consistency_lora")).toBe("false");
  });
  it("limits Consistency to Quality edits and ignores refinements in Generate", async () => {
    const settings = { ...defaultSettings, prompt: "weave", consistency_lora: true, detail_crop: true, specialized_outpaint: true };
    await submitJob(input({ operation: "edit", settings }));
    expect(vi.mocked(editWholeImage).mock.calls[0][1].consistency_lora).toBe(true);
    await submitJob(input({ operation: "edit", settings: { ...settings, model_name: "flux2-klein-4b-nvfp4" } }));
    expect(vi.mocked(editWholeImage).mock.calls[1][1].consistency_lora).toBe(false);
    await submitJob(input({ operation: "generate", settings }));
    const parameters = vi.mocked(generateImage).mock.calls[0][0];
    expect(parameters.model_name).toBe("flux2-klein-9b-nvfp4");
    expect(parameters).not.toHaveProperty("consistency_lora");
    expect(parameters).not.toHaveProperty("specialized_outpaint");
    expect(parameters).not.toHaveProperty("detail_crop");
  });
});
