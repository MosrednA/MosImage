import { ImageTask } from "../api";
import { CanvasTool } from "../canvas/tools";

export type Operation = "generate" | "replace" | "remove" | "expand";

export function taskFor(operation: Operation): ImageTask {
  return operation === "generate" ? "text-to-image" : operation === "expand" ? "outpaint" : "inpaint";
}

export function submitLabel(operation: Operation): string {
  return operation === "remove" ? "Remove" : operation === "expand" ? "Expand" : "Generate";
}

/** "Anna", "Anna and Tom", "Anna, Tom and Mia". */
export function joinNames(names: readonly string[]): string {
  return names.length < 2 ? names[0] ?? "" : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

export function promptPlaceholder(operation: Operation, hasImage: boolean, referenceNames: readonly string[] = []): string {
  if (operation === "remove") return "Optional: describe the background to restore…";
  if (operation === "expand") return "Describe the scene or what to add beyond the edges…";
  if (referenceNames.length > 1) return `What are ${joinNames(referenceNames)} doing here? e.g. sitting together on the bench`;
  if (referenceNames.length) return `What is ${referenceNames[0]} doing here? e.g. sitting on the bench, reading`;
  return hasImage ? "What would you like to see in your selection?" : "Describe the image you want to create…";
}

export function operationHint(operation: Operation, hasImage: boolean, hasMask: boolean, tool: CanvasTool, referenceNames: readonly string[] = []): string {
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
