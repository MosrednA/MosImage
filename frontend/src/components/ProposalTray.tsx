interface Props {
  thumbnails: string[];
  selected: number;
  onSelect: (index: number) => void;
  canCompare: boolean;
  comparing: boolean;
  onCompare: () => void;
  busy: boolean;
  onDiscard: () => void;
  onApply: () => void;
}

export function ProposalTray({ thumbnails, selected, onSelect, canCompare, comparing, onCompare, busy, onDiscard, onApply }: Props) {
  return <section className="proposal-tray" aria-label="Proposals">
    <div className="proposal-intro"><strong>Choose a proposal</strong><span>{selected + 1} of {thumbnails.length}</span></div>
    <div className="proposal-thumbnails">
      {thumbnails.map((url, index) => <button key={url} className={index === selected ? "selected" : ""} aria-pressed={index === selected} aria-label={"Proposal " + (index + 1)} onClick={() => onSelect(index)}>
        <img crossOrigin="anonymous" src={url} alt="" /><span>{index + 1}</span>
      </button>)}
    </div>
    <div className="proposal-actions">
      {canCompare && <button aria-pressed={comparing} onClick={onCompare}>{comparing ? "Show proposal" : "View original"}</button>}
      <button disabled={busy} onClick={onDiscard}>Discard</button>
      <button className="primary" disabled={busy} onClick={onApply}>Apply</button>
    </div>
  </section>;
}
