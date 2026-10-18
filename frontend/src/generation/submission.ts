import { EditParameters, Expansion, FaceBox, JobResponse, LayerRecord, editWholeImage, enhanceImage, fileUrl, fixFaces, generateImage, inpaintImage, outpaintImage } from "../api";
import { ImageSize } from "../canvas/coordinates";
import { validExpansion } from "../canvas/expansion";
import { sketchMask, sketchedPicture } from "../canvas/sketch";
import { EditorSettings, qualityOptions } from "../components/EditorSettings";
import { canvasBlob, loadImage, maskedLayer, shownMask } from "../layers/bitmaps";
import { MAX_LAYERS } from "../layers/stack";
import { ENHANCE_DETAIL, EditScope, EnhanceLevel, FaceMode, Operation, SketchScope, taskFor } from "./operation";

/** A full generation, the selected footprint of a regional AI edit, an edit of one layer, or the fixed faces. */
export type LayerResult = { kind: "selection"; mask: Blob; name: string; replaces: string | null }
  | { kind: "generation"; name: string; diffMask?: boolean }
  | { kind: "layer"; layerId: string; name: string }
  | { kind: "faces"; name: string };

export interface JobExtras {
  /** An object selection on this layer cuts it out, instead of selecting on the background. */
  layerId?: string;
  /** Undo label for that layer mask change. */
  maskLabel?: string;
  /** Expand adds canvas on the left and top; layers move along with the content. */
  layerOffset?: { x: number; y: number };
  /** The accepted generation or regional edit becomes a layer. */
  result?: LayerResult;
}

/** A queued AI job and how its result comes back into the document. */
export interface Submission {
  created: JobResponse;
  /** Expand proposes new revisions; everything else comes back as a layer or a new image. */
  operation: "outpaint" | null;
  extras: JobExtras;
}

export interface SubmissionInput {
  operation: Operation;
  settings: EditorSettings;
  referenceIds: number[];
  source: File | null;
  size: ImageSize | null;
  hasDocument: boolean;
  layerCount: number;
  mask: Blob | null;
  sketch: Blob | null;
  sketchScope: SketchScope;
  editScope: EditScope;
  faceMode: FaceMode;
  /** The faces picked on the canvas; without them the engine finds the faces, inside the selection if there is one. */
  faces: FaceBox[] | null;
  enhanceLevel: EnhanceLevel;
  /** The layer Edit works on alone. */
  editLayer: LayerRecord | null;
  /** The layer a regional edit blends into the scene. */
  blendLayer: LayerRecord | null;
  expansion: Expansion;
  /** What the AI works on: the picture as you see it, with its visible layers. */
  picture: () => Promise<File>;
}

/** Blending keeps most of the layer and redraws its light and edges. */
export const BLEND_STRENGTH = 0.6;

export function requestPrompt(operation: Operation, settings: EditorSettings): string {
  const prompt = settings.prompt.trim();
  if (operation !== "remove") return prompt;
  return prompt || "Continue the surrounding background.";
}

export function requestParameters(settings: EditorSettings, prompt: string): EditParameters {
  return {
    prompt, model_name: settings.model_name, runtime_mode: settings.runtime_mode,
    seed: settings.seed ? Number(settings.seed) : undefined, steps: settings.steps, guidance: settings.guidance, count: settings.count,
  };
}

/** A short layer name for an AI result. */
export function resultName(blended: string | null, operation: Operation, prompt: string): string {
  if (blended) return `${blended} blended`;
  if (operation === "remove") return "AI removal";
  const words = prompt.split(/\s+/).filter(Boolean).slice(0, 5).join(" ");
  return words ? `AI · ${words}` : "AI edit";
}

interface InpaintOptions { strength: number; editOperation: "replace" | "remove" | "sketch"; contextPadding: number }

function inpaintForm(parameters: EditParameters, image: File, mask: Blob, { strength, editOperation, contextPadding }: InpaintOptions): FormData {
  const form = new FormData();
  for (const [key, value] of Object.entries(parameters)) if (value !== undefined) form.set(key, String(value));
  form.set("strength", String(strength));
  form.set("image", image); form.set("mask", mask, "mask.png");
  form.set("edit_operation", editOperation);
  // The canvas sends the final soft mask (Expand/Soften), so no extra server-side ramp.
  form.set("blend_feather", "0");
  form.set("context_padding", String(contextPadding));
  return form;
}

/** Sends the current operation to the engine. */
export async function submitJob(input: SubmissionInput): Promise<Submission> {
  const { operation, settings, referenceIds, source, size, mask } = input;
  const task = taskFor(operation);
  const prompt = requestPrompt(operation, settings);
  const parameters = requestParameters(settings, prompt);
  const refinements = qualityOptions(settings, operation, !!input.blendLayer);
  const withReferences = { ...parameters, reference_ids: referenceIds };
  if (input.hasDocument && task !== "outpaint" && input.layerCount >= MAX_LAYERS) throw new Error(`A document can have up to ${MAX_LAYERS} layers. Delete one before generating.`);
  const asLayer = (created: JobResponse, result: LayerResult): Submission => ({ created, operation: null, extras: { result } });

  if (operation === "sketch") {
    if (!source || !input.hasDocument || !size) throw new Error("Open an image first.");
    const footprint = input.sketch && await sketchMask(input.sketch, size);
    if (!input.sketch || !footprint) throw new Error("Sketch something first: paint rough shapes with the Sketch tool.");
    const picture = await sketchedPicture(await input.picture(), input.sketch, size);
    const name = resultName(null, operation, prompt);
    // Picked references are drawn where the sketch shows them, in its pose and size.
    if (input.sketchScope === "whole") return asLayer(await editWholeImage(picture, { ...withReferences, mode: "sketch" }), { kind: "generation", name });
    // Only the sketch and a margin around it are redrawn; the rest of the picture stays exact.
    const form = inpaintForm(parameters, picture, footprint, { strength: 1, editOperation: "sketch", contextPadding: settings.context_padding });
    for (const id of referenceIds) form.append("reference_ids", String(id));
    return asLayer(await inpaintImage(form), { kind: "selection", mask: footprint, name, replaces: null });
  }
  if (operation === "faces") {
    if (!source || !input.hasDocument) throw new Error("Open an image first.");
    const { faces } = input;
    if (faces && !faces.length) throw new Error("Click a face on the picture to fix it.");
    // Picked faces already account for the selection; otherwise the selection limits which faces are fixed.
    const created = await fixFaces(await input.picture(), { ...withReferences, faithful: input.faceMode === "faithful", faces: faces ?? undefined }, faces ? null : mask);
    return asLayer(created, { kind: "faces", name: "Fixed faces" });
  }
  if (operation === "enhance") {
    if (!source || !input.hasDocument) throw new Error("Open an image first.");
    // Without a selection the whole picture is enhanced and covers the canvas.
    const created = await enhanceImage(await input.picture(), { ...parameters, model_name: refinements.model_name, detail: ENHANCE_DETAIL[input.enhanceLevel] }, mask);
    return asLayer(created, mask ? { kind: "selection", mask, name: "Enhanced", replaces: null } : { kind: "generation", name: "Enhanced" });
  }
  if (task === "text-to-image") {
    const created = await generateImage({ ...withReferences, width: settings.width, height: settings.height });
    if (!input.hasDocument) return { created, operation: null, extras: {} };
    return asLayer(created, { kind: "generation", name: resultName(null, operation, prompt) });
  }
  if (task === "edit") {
    withReferences.consistency_lora = refinements.consistency_lora;
    if (!source || !input.hasDocument) throw new Error("Open an image first.");
    const layer = input.editLayer;
    if (layer) {
      // Only this layer, through its mask; the result replaces it in place.
      const image = await loadImage(fileUrl(layer.image_url));
      const layerMask = await shownMask(layer);
      const picture = new File([await canvasBlob(maskedLayer(image, layerMask, layer.width, layer.height))], "layer.png", { type: "image/png" });
      return asLayer(await editWholeImage(picture, withReferences), { kind: "layer", layerId: layer.id, name: `${layer.name} · AI` });
    }
    if (input.editScope !== "visible") throw new Error("Pick the layer to edit in the Layers panel.");
    // The AI sees every visible layer; local changes get a difference mask so untouched background stays sharp.
    return asLayer(await editWholeImage(await input.picture(), withReferences), { kind: "generation", name: resultName(null, operation, prompt), diffMask: true });
  }
  if (task === "outpaint") {
    if (!source || !size || !validExpansion(size, input.expansion)) throw new Error("Choose a valid canvas size first.");
    const { expansion } = input;
    // The new edges continue the picture as you see it; the engine keeps the layers out of the background.
    return { created: await outpaintImage(await input.picture(), { ...parameters, ...expansion,
      model_name: refinements.model_name, consistency_lora: refinements.consistency_lora,
      specialized_outpaint: refinements.specialized_outpaint,
      steps: refinements.specialized_outpaint ? 4 : parameters.steps,
      guidance: refinements.specialized_outpaint ? 1 : parameters.guidance,
    }), operation: "outpaint", extras: { layerOffset: { x: expansion.left, y: expansion.top } } };
  }
  if (!source || !mask || !input.hasDocument) throw new Error("Select the area you want to change first.");
  // Every regional edit becomes a layer over the picture the AI saw.
  const blendLayer = input.blendLayer;
  const form = inpaintForm(parameters, await input.picture(), mask, {
    strength: refinements.detail_crop ? 1 : blendLayer ? Math.min(settings.strength, BLEND_STRENGTH) : settings.strength,
    editOperation: operation === "remove" ? "remove" : "replace",
    contextPadding: settings.context_padding,
  });
  form.set("consistency_lora", String(refinements.consistency_lora));
  form.set("detail_crop", String(refinements.detail_crop));
  form.set("scene_context", settings.scene_context);
  for (const id of referenceIds) form.append("reference_ids", String(id));
  return asLayer(await inpaintImage(form), { kind: "selection", mask, name: resultName(blendLayer?.name ?? null, operation, prompt), replaces: blendLayer?.id ?? null });
}
