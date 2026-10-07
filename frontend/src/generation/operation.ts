import { ImageTask } from "../api";
import { CanvasTool } from "../canvas/tools";

export type Operation = "generate" | "replace" | "remove" | "expand";

export function taskFor(operation: Operation): ImageTask {
  return operation === "generate" ? "text-to-image" : operation === "expand" ? "outpaint" : "inpaint";
}

export function submitLabel(operation: Operation): string {
  return operation === "remove" ? "Remove" : operation === "expand" ? "Expand" : "Generate";
}

export function promptPlaceholder(operation: Operation, hasImage: boolean): string {
  if (operation === "remove") return "Optional: describe the background to restore…";
  if (operation === "expand") return "Describe the scene or what to add beyond the edges…";
  return hasImage ? "What would you like to see in your selection?" : "Describe the image you want to create…";
}

export function operationHint(operation: Operation, hasImage: boolean, hasMask: boolean, tool: CanvasTool): string {
  if (operation === "expand") return "Describe the scene or new content. Your original image stays intact.";
  if (!hasImage) return "Describe a new image, or open one to edit.";
  if (hasMask) {
    return operation === "remove"
      ? "Include the object's shadow and glow. The soft edge blends the fill into its surroundings."
      : "Describe what should appear. Leave room in the selection for the whole subject and its shadow.";
  }
  return tool === "object" ? "Click an object to select it. Right-click to exclude a part." : "Paint or draw the area you want to change.";
}
