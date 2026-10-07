import { IconName } from "../components/Icon";

export type MaskTool = "brush" | "eraser" | "rectangle" | "lasso";
export type CanvasTool = MaskTool | "move" | "object";

export const MASK_TOOLS: readonly MaskTool[] = ["brush", "eraser", "rectangle", "lasso"];

export function isMaskTool(tool: CanvasTool): tool is MaskTool {
  return (MASK_TOOLS as readonly CanvasTool[]).includes(tool);
}

interface ToolInfo { label: string; icon: IconName; key: string; status: string }

export const TOOLS: Record<CanvasTool, ToolInfo> = {
  object: { label: "Object", icon: "object", key: "O", status: "Click to add · Right-click to exclude · Click a marker to remove it" },
  brush: { label: "Brush", icon: "brush", key: "B", status: "Paint to select · Alt subtracts · [ ] size · Space pans" },
  eraser: { label: "Eraser", icon: "eraser", key: "E", status: "Paint to remove from the selection · [ ] size" },
  lasso: { label: "Lasso", icon: "lasso", key: "L", status: "Draw around an area · Alt subtracts" },
  rectangle: { label: "Box", icon: "rectangle", key: "M", status: "Drag a box · Alt subtracts" },
  move: { label: "Pan", icon: "move", key: "V", status: "Drag to pan · Ctrl + scroll zooms" },
};

/** Rail order: selection tools first, navigation last. */
export const RAIL_TOOLS: readonly CanvasTool[] = ["object", "brush", "eraser", "lasso", "rectangle"];

export function toolForKey(key: string): CanvasTool | null {
  const match = (Object.keys(TOOLS) as CanvasTool[]).find(tool => TOOLS[tool].key.toLowerCase() === key.toLowerCase());
  return match ?? null;
}
