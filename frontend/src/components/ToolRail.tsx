import { CanvasTool, RAIL_TOOLS, TOOLS } from "../canvas/tools";
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
  const toolButton = (candidate: CanvasTool) => {
    const info = TOOLS[candidate];
    const active = hasImage && !expanding && tool === candidate;
    return <button key={candidate} className={active ? "active" : ""} aria-pressed={active} title={`${info.label} (${info.key})`} disabled={!hasImage || locked} onClick={() => onTool(candidate)}>
      <Icon name={info.icon} /><span>{info.label}</span>
    </button>;
  };
  return <nav className="tool-rail" aria-label="Tools">
    <button className={!hasImage ? "active" : ""} title="New image" disabled={locked} onClick={onNew}><Icon name="spark" /><span>New</span></button>
    <div className="rail-divider" />
    {RAIL_TOOLS.map(toolButton)}
    <div className="rail-divider" />
    {toolButton("sketch")}
    {toolButton("transform")}
    <button className={expanding ? "active" : ""} aria-pressed={expanding} title="Expand canvas (C)" disabled={!hasImage || locked} onClick={onExpand}><Icon name="expand" /><span>Expand</span></button>
    {toolButton("move")}
  </nav>;
}
