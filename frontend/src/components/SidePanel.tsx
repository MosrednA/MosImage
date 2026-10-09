import { ReactNode } from "react";
import { Icon } from "./Icon";

export type SideTab = "layers" | "references";

interface Props {
  tab: SideTab;
  onTab: (tab: SideTab) => void;
  collapsed: boolean;
  onCollapsed: (collapsed: boolean) => void;
  layerCount: number;
  referenceCount: number;
  /** False on the start screen, where only references apply. */
  showLayers?: boolean;
  children: ReactNode;
}

/** The right-hand panel: image layers and references, which stay separate. */
export function SidePanel({ tab, onTab, collapsed, onCollapsed, layerCount, referenceCount, showLayers = true, children }: Props) {
  const open = (next: SideTab) => { onTab(next); onCollapsed(false); };
  if (collapsed) {
    return <aside className="side-panel collapsed" aria-label="Layers and references">
      {showLayers && <button className="side-expand" title="Show layers" aria-label={`Show layers (${layerCount})`} onClick={() => open("layers")}>
        <Icon name="layers" />{layerCount > 0 && <span className="side-count">{layerCount}</span>}
      </button>}
      <button className="side-expand" title="Show references" aria-label={`Show references (${referenceCount})`} onClick={() => open("references")}>
        <Icon name="person" />{referenceCount > 0 && <span className="side-count">{referenceCount}</span>}
      </button>
    </aside>;
  }
  return <aside className="side-panel" aria-label="Layers and references">
    <div className="side-tabs" role="tablist">
      {showLayers && <button role="tab" aria-selected={tab === "layers"} onClick={() => onTab("layers")}>
        <Icon name="layers" />Layers{layerCount > 0 && <span className="side-count">{layerCount}</span>}
      </button>}
      <button role="tab" aria-selected={tab === "references"} onClick={() => onTab("references")}>
        <Icon name="person" />References{referenceCount > 0 && <span className="side-count">{referenceCount}</span>}
      </button>
      <button className="icon-button" title="Hide panel" aria-label="Hide panel" onClick={() => onCollapsed(true)}><Icon name="chevronRight" /></button>
    </div>
    {children}
  </aside>;
}
