import { IconName } from "../components/Icon";
import { MOD } from "../platform";

export type MaskTool = "brush" | "eraser" | "rectangle" | "lasso";
export type CanvasTool = MaskTool | "move" | "object" | "transform" | "sketch";

export const MASK_TOOLS: readonly MaskTool[] = ["brush", "eraser", "rectangle", "lasso"];

export function isMaskTool(tool: CanvasTool): tool is MaskTool {
  return (MASK_TOOLS as readonly CanvasTool[]).includes(tool);
}

/** Tools that make a selection on the picture, or edit a layer's mask while masking it. */
export function isSelectionTool(tool: CanvasTool): boolean {
  return tool === "object" || isMaskTool(tool);
}

interface ToolInfo { label: string; icon: IconName; key: string; status: string }

/** Keys follow Photoshop: W quick selection, V move, H hand, M marquee, L lasso, C crop. */
export const TOOLS: Record<CanvasTool, ToolInfo> = {
  object: { label: "Object", icon: "object", key: "W", status: "Click to add · Right-click to exclude · Click a marker to remove it" },
  brush: { label: "Brush", icon: "brush", key: "B", status: "Paint to select · Alt subtracts · [ ] size · Space pans" },
  eraser: { label: "Eraser", icon: "eraser", key: "E", status: "Paint to remove from the selection · [ ] size" },
  lasso: { label: "Lasso", icon: "lasso", key: "L", status: "Draw around an area · Alt subtracts" },
  rectangle: { label: "Box", icon: "rectangle", key: "M", status: "Drag a box · Alt subtracts" },
  sketch: { label: "Sketch", icon: "pencil", key: "P", status: "Paint rough shapes in colour · Alt erases · [ ] size" },
  transform: { label: "Transform", icon: "transform", key: "V", status: "Click a layer to select it · Drag to move · Handles scale, stretch and rotate" },
  move: { label: "Pan", icon: "move", key: "H", status: `Drag to pan · ${MOD} + scroll zooms` },
};

/** What the tools do while masking a layer: they edit its mask instead of the AI selection. */
export const LAYER_STATUS: Record<CanvasTool, string> = {
  object: "Click the part of the layer to keep · Right-click to exclude",
  brush: "Paint the mask: white shows, black hides, grey between · X swaps · Alt hides · [ ] size · \\ shows hidden parts",
  eraser: "Paint to hide parts of the layer · Alt restores · [ ] size · \\ shows hidden parts",
  lasso: "Draw around an area to hide or restore it · Alt switches",
  rectangle: "Drag a box to hide or restore it · Alt switches",
  sketch: "Paint rough shapes in colour · Alt erases · [ ] size",
  transform: "Drag to move · Corners scale (Shift stretches) · Edges stretch · Shift snaps rotation",
  move: `Drag to pan · ${MOD} + scroll zooms`,
};

/** Rail order: selection tools first, navigation last. */
export const RAIL_TOOLS: readonly CanvasTool[] = ["object", "brush", "eraser", "lasso", "rectangle"];

export function toolForKey(key: string): CanvasTool | null {
  const match = (Object.keys(TOOLS) as CanvasTool[]).find(tool => TOOLS[tool].key.toLowerCase() === key.toLowerCase());
  return match ?? null;
}
