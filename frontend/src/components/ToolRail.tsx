import { CanvasTool } from "../canvas/tools";
import { Icon } from "./Icon";

interface Props {
  hasImage: boolean;
  tool: CanvasTool;
  expanding: boolean;
  locked: boolean;
  onNew: () => void;
  onTool: (tool: CanvasTool) => void;
  onExpand: () => void;
}

export function ToolRail({ hasImage, tool, expanding, locked, onNew, onTool, onExpand }: Props) {
  const active = (candidate: boolean) => hasImage && candidate && !expanding ? "active" : "";
  return <nav className="tool-rail" aria-label="Tools">
    <button className={!hasImage ? "active" : ""} title="New image" disabled={locked} onClick={onNew}><Icon name="spark" /><span>New</span></button>
    <div className="rail-divider" />
    <button className={active(tool === "object")} title="Select object (O)" disabled={!hasImage || locked} onClick={() => onTool("object")}><Icon name="object" /><span>Object</span></button>
    <button className={active(["brush", "eraser", "rectangle", "lasso"].includes(tool))} title="Brush (B)" disabled={!hasImage || locked} onClick={() => onTool("brush")}><Icon name="brush" /><span>Brush</span></button>
    <button className={expanding ? "active" : ""} title="Expand image (E)" disabled={!hasImage || locked} onClick={onExpand}><Icon name="expand" /><span>Expand</span></button>
    <button className={active(tool === "move")} title="Pan (V)" disabled={!hasImage || locked} onClick={() => onTool("move")}><Icon name="move" /><span>Pan</span></button>
    <div className="rail-footer"><span>LOCAL<br />STUDIO</span></div>
  </nav>;
}
