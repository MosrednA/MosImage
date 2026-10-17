interface Props {
  thumbnails: string[];
  selected: number;
  onSelect: (index: number) => void;
  canCompare: boolean;
  comparing: boolean;
  onCompare: () => void;
  splitting?: boolean;
  onSplit?: () => void;
  busy: boolean;
  onDiscard: () => void;
  onApply: () => void;
  /** Applies the proposal as a layer and immediately activates the layer mask brush to restore or hide parts. */
  onApplyAndMask?: () => void;
  /** Applies every proposal as a layer of its own, when results arrive as layers. */
  onApplySeparately?: () => void;
}

export function ProposalTray({ thumbnails, selected, onSelect, canCompare, comparing, onCompare, splitting, onSplit, busy, onDiscard, onApply, onApplyAndMask, onApplySeparately }: Props) {
  const several = thumbnails.length > 1;
  return <section className="proposal-tray" aria-label="Proposals">
    <div className="proposal-intro"><strong>Choose a proposal</strong><span>{selected + 1} of {thumbnails.length}</span></div>
    <div className="proposal-thumbnails">
      {thumbnails.map((url, index) => <button key={url} className={index === selected ? "selected" : ""} aria-pressed={index === selected} aria-label={"Proposal " + (index + 1)} onClick={() => onSelect(index)}>
        <img crossOrigin="anonymous" src={url} alt="" /><span>{index + 1}</span>
      </button>)}
    </div>
    <div className="proposal-actions">
      {canCompare && <button aria-pressed={comparing} onClick={onCompare}>{comparing ? "Show proposal" : "View original"}</button>}
      {canCompare && onSplit && <button aria-pressed={!!splitting} title="Drag a split slider across the canvas to compare before and after" onClick={onSplit}>Split compare</button>}
      <button disabled={busy} onClick={onDiscard}>Discard</button>
      {several && onApplySeparately && <button disabled={busy} title="Every proposal on a layer of its own; the others start hidden" onClick={onApplySeparately}>All as layers</button>}
      {onApplyAndMask && <button disabled={busy} title="Apply as a layer and immediately brush its mask to restore or hide parts" onClick={onApplyAndMask}>Apply &amp; mask</button>}
      <button className="primary" disabled={busy} title={several && onApplySeparately ? "One layer that keeps every proposal: switch them later with , and ." : undefined} onClick={onApply}>Apply</button>
    </div>
  </section>;
}
