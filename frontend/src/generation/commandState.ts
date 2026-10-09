import { CanvasTool } from "../canvas/tools";
import { EditScope, Operation, SketchScope, facesHint, isWholeImage, layerEditHint, needsQuality, operationHint, sketchHint, taskFor } from "./operation";

/** What the command dock needs to know to enable Submit and explain the next step. */
export interface CommandState {
  operation: Operation;
  prompt: string;
  /** The Quality model (FLUX.2 Klein) is chosen. */
  quality: boolean;
  hasImage: boolean;
  hasMask: boolean;
  hasSketch: boolean;
  expansionValid: boolean;
  editScope: EditScope;
  editLayerName: string | null;
  sketchScope: SketchScope;
  referenceNames: string[];
  tool: CanvasTool;
  /** The layer the canvas works on, and whether its mask is being edited. */
  activeLayerName: string | null;
  layerMasking: boolean;
  /** The layer a regional edit blends in. */
  blendingName: string | null;
}

export function canSubmit(state: CommandState): boolean {
  const prompted = !!state.prompt.trim();
  const task = taskFor(state.operation);
  if (state.operation === "sketch") return state.hasImage && state.hasSketch && prompted && state.quality;
  if (state.operation === "faces") return state.hasImage && state.quality;
  if (task === "outpaint") return state.expansionValid && prompted;
  if (state.referenceNames.length && !state.quality) return false;
  const area = task === "text-to-image" || (task === "edit" && state.hasImage && (state.editScope === "visible" || state.editLayerName !== null)) || state.hasMask;
  return (state.operation === "remove" || prompted) && area;
}

export function commandHint(state: CommandState): string {
  const { operation, referenceNames } = state;
  if (state.activeLayerName !== null && state.layerMasking) return `Masking layer “${state.activeLayerName}”: the tools hide or restore parts of it. Press Esc when done.`;
  if (state.activeLayerName !== null) return `Editing layer “${state.activeLayerName}”. Pick a selection tool or press Esc for AI edits; they see every visible layer.`;
  if (referenceNames.length && !state.quality) return "References need the Quality model. Choose Quality in generation settings.";
  if (operation === "edit" && state.editScope === "layer") return layerEditHint(state.editLayerName, referenceNames);
  if (needsQuality(operation) && !state.quality) return `${operation === "sketch" ? "Sketches need" : "Fix faces needs"} the Quality model. Choose Quality in generation settings.`;
  if (operation === "sketch") return sketchHint(state.sketchScope, state.hasSketch);
  if (operation === "faces") return facesHint(state.hasMask, referenceNames);
  if (state.blendingName !== null && state.hasMask) return `Blending “${state.blendingName}” in: the AI redraws it and its edges to fit the scene. Widen the area with the brush if needed.`;
  const asLayer = state.hasImage && operation !== "expand" && !isWholeImage(operation) ? " Apply adds the result as a new layer." : "";
  return operationHint(operation, state.hasImage, state.hasMask, state.tool, referenceNames) + asLayer;
}
