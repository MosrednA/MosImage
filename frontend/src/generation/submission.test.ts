import { beforeEach, describe, expect, it, vi } from "vitest";
import { editWholeImage, fixFaces, generateImage, inpaintImage, outpaintImage } from "../api";
import { defaultSettings } from "../components/EditorSettings";
import { MAX_LAYERS } from "../layers/stack";
import { BLEND_STRENGTH, SubmissionInput, requestPrompt, resultName, submitJob } from "./submission";

vi.mock("../api", async original => {
  const created = { id: "job", status: "queued" };
  return {
    ...await original<typeof import("../api")>(),
    editWholeImage: vi.fn(async () => created), fixFaces: vi.fn(async () => created), generateImage: vi.fn(async () => created),
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
    hasDocument: true, layerCount: 0, mask: new Blob(["mask"]), sketch: null, sketchScope: "strokes", editScope: "visible", editLayer: null, blendLayer: null,
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
  it("limits fixed faces to the selection", async () => {
    const mask = new Blob(["faces"]);
    const submission = await submitJob(input({ operation: "faces", mask, referenceIds: [7] }));
    expect(vi.mocked(fixFaces).mock.calls[0][1].reference_ids).toEqual([7]);
    expect(vi.mocked(fixFaces).mock.calls[0][2]).toBe(mask);
    expect(submission.extras.result).toEqual({ kind: "faces", name: "Fixed faces" });
  });
  it("keeps start-screen references with a new image", async () => {
    const start = await submitJob(input({ operation: "generate", hasDocument: false, source: null, referenceIds: [5] }));
    expect(start.startReferences).toEqual([5]);
    expect(start.extras.result).toBeUndefined();
    const inDocument = await submitJob(input({ operation: "generate", referenceIds: [5] }));
    expect(inDocument.startReferences).toEqual([]);
    expect(inDocument.extras.result?.kind).toBe("generation");
    expect(vi.mocked(generateImage).mock.calls[0][0]).toMatchObject({ width: 1024, height: 1024, reference_ids: [5] });
  });
  it("needs a layer for a layer edit", async () => {
    await expect(submitJob(input({ operation: "edit", editScope: "layer" }))).rejects.toThrow("Pick the layer");
    const whole = await submitJob(input({ operation: "edit" }));
    expect(whole.extras.result?.kind).toBe("generation");
  });
  it("moves layers along when expanding", async () => {
    const submission = await submitJob(input({ operation: "expand", expansion: { left: 64, top: 32, right: 0, bottom: 0 } }));
    expect(vi.mocked(outpaintImage)).toHaveBeenCalled();
    expect(submission).toMatchObject({ operation: "outpaint", extras: { layerOffset: { x: 64, y: 32 } } });
  });
  it("refuses another layer at the limit", async () => {
    await expect(submitJob(input({ layerCount: MAX_LAYERS }))).rejects.toThrow(`up to ${MAX_LAYERS} layers`);
  });
});
