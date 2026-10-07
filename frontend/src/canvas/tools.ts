import { IconName } from "../components/Icon";

export type MaskTool = "brush" | "eraser" | "rectangle" | "lasso";
export type CanvasTool = MaskTool | "move" | "object" | "transform";

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
  transform: { label: "Transform", icon: "transform", key: "T", status: "Click a layer to select it · Drag to move · Handles scale, stretch and rotate" },
  move: { label: "Pan", icon: "move", key: "V", status: "Drag to pan · Ctrl + scroll zooms" },
};

/** What the tools do when a layer is selected: they edit its mask instead of the AI selection. */
export const LAYER_STATUS: Record<CanvasTool, string> = {
  object: "Click the part of the layer to keep · Right-click to exclude",
  brush: "Paint to bring hidden parts of the layer back · [ ] size",
  eraser: "Paint to hide parts of the layer · [ ] size",
  lasso: "Draw around an area to hide or restore it · Alt switches",
  rectangle: "Drag a box to hide or restore it · Alt switches",
  transform: "Drag to move · Corners scale (Shift stretches) · Edges stretch · Shift snaps rotation",
  move: "Drag to pan · Ctrl + scroll zooms",
};

/** Rail order: selection tools first, navigation last. */
export const RAIL_TOOLS: readonly CanvasTool[] = ["object", "brush", "eraser", "lasso", "rectangle"];

export function toolForKey(key: string): CanvasTool | null {
  const match = (Object.keys(TOOLS) as CanvasTool[]).find(tool => TOOLS[tool].key.toLowerCase() === key.toLowerCase());
  return match ?? null;
}
