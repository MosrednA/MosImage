import { ImageTask } from "../api";
import { CanvasTool } from "../canvas/tools";

export type Operation = "generate" | "edit" | "replace" | "remove" | "expand" | "sketch" | "faces";

/** Sketch and Fix faces run on FLUX.2 Klein's edit pipeline, so they count as edits. */
export function taskFor(operation: Operation): ImageTask {
  return operation === "generate" ? "text-to-image" : operation === "edit" || operation === "sketch" || operation === "faces" ? "edit" : operation === "expand" ? "outpaint" : "inpaint";
}

/** Render a sketch on its own patch of the picture, or redraw the whole picture around it. */
export type SketchScope = "strokes" | "whole";

/** Operations that work on the whole picture, without a selection or an active layer. */
export function isWholeImage(operation: Operation): boolean {
  return operation === "generate" || operation === "edit";
}

/** Edit works on everything visible, or on one layer alone. */
export type EditScope = "visible" | "layer";

export function layerEditHint(layerName: string | null, referenceNames: readonly string[] = []): string {
  if (!layerName) return "Pick the layer to edit in the Layers panel.";
  const keep = referenceNames.length ? ` ${joinNames(referenceNames)} ${referenceNames.length > 1 ? "keep" : "keeps"} the look of their reference.` : "";
  return `The AI edits only “${layerName}”; its transparent parts stay transparent.${keep} Apply puts the result above it and hides the original.`;
}

export function submitLabel(operation: Operation): string {
  return operation === "remove" ? "Remove" : operation === "expand" ? "Expand" : operation === "edit" ? "Edit"
    : operation === "sketch" ? "Render" : operation === "faces" ? "Fix faces" : "Generate";
}

export function sketchHint(scope: SketchScope, hasSketch: boolean): string {
  if (!hasSketch) return "Paint rough shapes in colour with the Sketch tool, then say what they are. Crude is fine: the AI follows shape, size and colour.";
  return scope === "strokes"
    ? "Render turns the strokes into the real thing; the rest of the picture stays as it is. Apply adds it as a new layer."
    : "Render redraws the whole picture around your sketch, so lighting and style match. Apply adds it as a new layer.";
}

export function facesHint(hasSelection: boolean, referenceNames: readonly string[] = []): string {
  const which = hasSelection ? "Faces inside your selection" : "Every face in the picture";
  const likeness = referenceNames.length ? ` ${joinNames(referenceNames)} ${referenceNames.length > 1 ? "keep" : "keeps"} the look of their reference.` : " Identity and expression stay the same.";
  return `${which} ${hasSelection ? "are" : "is"} redrawn sharper and pasted back.${likeness} Apply adds the faces as a new layer.`;
}

/** "Anna", "Anna and Tom", "Anna, Tom and Mia". */
export function joinNames(names: readonly string[]): string {
  return names.length < 2 ? names[0] ?? "" : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

export function promptPlaceholder(operation: Operation, hasImage: boolean, referenceNames: readonly string[] = []): string {
  if (operation === "generate") {
    if (referenceNames.length) return `Where ${referenceNames.length > 1 ? "are" : "is"} ${joinNames(referenceNames)}? e.g. riding a roller coaster together`;
    return hasImage ? "Describe a new image to add as a layer…" : "Describe the image you want to create…";
  }
  if (operation === "edit") {
    if (referenceNames.length) return `What should change? e.g. put ${joinNames(referenceNames)} next to the fountain`;
    return "How should the picture change? e.g. make it a snowy night";
  }
  if (operation === "remove") return "Optional: describe the background to restore…";
  if (operation === "sketch") return "What did you draw? e.g. a deer drinking at the water";
  if (operation === "faces") return "Optional: anything extra, e.g. eyes open, gentle smile";
  if (operation === "expand") return "Describe the scene or what to add beyond the edges…";
  if (referenceNames.length > 1) return `What are ${joinNames(referenceNames)} doing here? e.g. sitting together on the bench`;
  if (referenceNames.length) return `What is ${referenceNames[0]} doing here? e.g. sitting on the bench, reading`;
  return hasImage ? "What would you like to see in your selection?" : "Describe the image you want to create…";
}

export function operationHint(operation: Operation, hasImage: boolean, hasMask: boolean, tool: CanvasTool, referenceNames: readonly string[] = []): string {
  if (operation === "generate") {
    const layer = hasImage ? " Apply adds it as a new layer, centred and fitted to your canvas." : "";
    if (referenceNames.length) return `Describe a new scene with ${joinNames(referenceNames)}; each keeps the look of their reference.${layer}`;
    return hasImage ? "Generate a full image." + layer : "Describe a new image, open one to edit, or add references to put people in a new scene.";
  }
  if (operation === "edit") {
    const keep = referenceNames.length
      ? `${joinNames(referenceNames)} ${referenceNames.length > 1 ? "keep" : "keeps"} the look of their reference.`
      : "Faces and everything you don't mention stay as they are.";
    return `The AI redraws the whole picture by your instruction. ${keep} Apply adds the result as a new layer.`;
  }
  if (operation === "expand") return "Describe the scene or new content. Your original image stays intact.";
  if (!hasImage) return "Describe a new image, or open one to edit.";
  if (referenceNames.length && operation === "replace") {
    const names = joinNames(referenceNames), several = referenceNames.length > 1;
    return hasMask
      ? `${names} ${several ? "appear" : "appears"} inside your selection, at the selection's size. Two proposals give you a choice.`
      : several
        ? `Select where ${names} should appear. Leave room for everyone, with their shadows.`
        : `Select where ${names} should appear. Leave room for the whole body and its shadow.`;
  }
  if (hasMask) {
    return operation === "remove"
      ? "Include the object's shadow and glow. The soft edge blends the fill into its surroundings."
      : "Describe what should appear. Leave room in the selection for the whole subject and its shadow.";
  }
  return tool === "object" ? "Click an object to select it. Right-click to exclude a part." : "Paint or draw the area you want to change.";
}
