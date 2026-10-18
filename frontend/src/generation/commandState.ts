import { CanvasTool } from "../canvas/tools";
import { FacePicks } from "./faces";
import { EditScope, FaceMode, Operation, SketchScope, enhanceHint, facesHint, isWholeImage, layerEditHint, operationHint, sketchHint, taskFor } from "./operation";

/** What the command dock needs to know to enable Submit and explain the next step. */
export interface CommandState {
  operation: Operation;
  prompt: string;
  hasImage: boolean;
  hasMask: boolean;
  hasSketch: boolean;
  expansionValid: boolean;
  editScope: EditScope;
  editLayerName: string | null;
  sketchScope: SketchScope;
  faceMode?: FaceMode;
  /** The faces shown on the canvas for Fix faces, once they are found. */
  facePicks?: FacePicks | null;
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
  if (state.operation === "sketch") return state.hasImage && state.hasSketch && prompted;
  if (state.operation === "faces") return state.hasImage && (state.facePicks?.picked ?? 1) > 0;
  if (state.operation === "enhance") return state.hasImage;
  if (task === "outpaint") return state.expansionValid && prompted;
  const area = task === "text-to-image" || (task === "edit" && state.hasImage && (state.editScope === "visible" || state.editLayerName !== null)) || state.hasMask;
  return (state.operation === "remove" || prompted) && area;
}

export function commandHint(state: CommandState): string {
  const { operation, referenceNames } = state;
  if (state.activeLayerName !== null && state.layerMasking) return `Painting the mask of “${state.activeLayerName}”: Brush paints its tone, white shows and black hides (X swaps), Eraser hides, \\ tints hidden parts red. Click the layer thumbnail or press Esc when done.`;
  if (state.activeLayerName !== null) return `Editing layer “${state.activeLayerName}”. Brush or another selection tool selects on the picture for AI edits, which see every visible layer; the Layers panel paints its mask.`;
  if (operation === "edit" && state.editScope === "layer") return layerEditHint(state.editLayerName, referenceNames);
  if (operation === "sketch") return sketchHint(state.sketchScope, state.hasSketch, referenceNames);
  if (operation === "faces") return facesHint(state.hasMask, referenceNames, state.faceMode, state.facePicks);
  if (operation === "enhance") return enhanceHint(state.hasMask);
  if (state.blendingName !== null && state.hasMask) return `Blending “${state.blendingName}” in: the AI redraws it and its edges to fit the scene. Widen the area with the brush if needed.`;
  const asLayer = state.hasImage && operation !== "expand" && !isWholeImage(operation) ? " Apply adds the result as a new layer." : "";
  return operationHint(operation, state.hasImage, state.hasMask, state.tool, referenceNames) + asLayer;
}
