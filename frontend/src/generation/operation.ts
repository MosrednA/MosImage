import { ImageTask } from "../api";
import { CanvasTool } from "../canvas/tools";

export type Operation = "generate" | "edit" | "replace" | "remove" | "expand";

export function taskFor(operation: Operation): ImageTask {
  return operation === "generate" ? "text-to-image" : operation === "edit" ? "edit" : operation === "expand" ? "outpaint" : "inpaint";
}

/** Operations that work on the whole picture, without a selection or an active layer. */
export function isWholeImage(operation: Operation): boolean {
  return operation === "generate" || operation === "edit";
}

export function submitLabel(operation: Operation): string {
  return operation === "remove" ? "Remove" : operation === "expand" ? "Expand" : operation === "edit" ? "Edit" : "Generate";
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
