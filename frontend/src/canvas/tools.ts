export type MaskTool = "brush" | "eraser" | "rectangle" | "lasso";
export type CanvasTool = MaskTool | "move" | "object";

export const MASK_TOOLS: readonly MaskTool[] = ["brush", "eraser", "rectangle", "lasso"];

export function isMaskTool(tool: CanvasTool): tool is MaskTool {
  return (MASK_TOOLS as readonly CanvasTool[]).includes(tool);
}

export function toolLabel(tool: CanvasTool): string {
  return { object: "Object", brush: "Brush", eraser: "Eraser", rectangle: "Rectangle", lasso: "Lasso", move: "Pan" }[tool];
}

export function toolStatus(tool: CanvasTool): string {
  if (tool === "object") return "Click an object · Alt + click to refine";
  if (tool === "move") return "Drag to pan · middle mouse works with any tool";
  if (tool === "brush") return "Paint your selection · hold Alt to subtract";
  if (tool === "eraser") return "Paint to erase selection";
  if (tool === "rectangle") return "Drag a rectangle · hold Alt to subtract";
  return "Draw around an area · hold Alt to subtract";
}
