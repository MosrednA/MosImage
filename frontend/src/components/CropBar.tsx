import { KeyboardEvent, useEffect, useState } from "react";
import { ImageSize } from "../canvas/coordinates";
import { CROP_RATIOS, CropRatio, CropRect, MAX_CROP, MIN_CROP, validCrop } from "../canvas/crop";

interface Props {
  size: ImageSize;
  rect: CropRect;
  ratio: CropRatio;
  busy: boolean;
  onRatio: (ratio: CropRatio) => void;
  onSize: (width: number, height: number, keep: "width" | "height") => void;
  onReset: () => void;
  onCancel: () => void;
  onApply: () => void;
}

const RATIO_LABELS: Record<CropRatio, string> = { free: "Free", original: "Original", "1:1": "1:1", "4:5": "4:5", "3:2": "3:2", "16:9": "16:9", "9:16": "9:16" };

/** A pixel field that applies on Enter or when you leave it, so typing passes through small numbers. */
function PixelField({ label, name, value, onCommit }: { label: string; name: string; value: number; onCommit: (value: number) => void }) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(value)), [value]);
  function commit() {
    const next = Math.round(Number(draft));
    if (Number.isFinite(next) && next >= MIN_CROP && next <= MAX_CROP && next !== value) onCommit(next);
    else setDraft(String(value));
  }
  function key(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter") { event.preventDefault(); event.currentTarget.blur(); }
    if (event.key === "Escape") { setDraft(String(value)); event.currentTarget.blur(); }
  }
  return <label><span>{label}</span><input type="number" aria-label={name} min={MIN_CROP} max={MAX_CROP} value={draft} onChange={event => setDraft(event.target.value)} onBlur={commit} onKeyDown={key} /></label>;
}

export function CropBar({ size, rect, ratio, busy, onRatio, onSize, onReset, onCancel, onApply }: Props) {
  return <section className="proposal-tray crop-bar" aria-label="Crop">
    <div className="proposal-intro"><strong>Crop canvas</strong><span>{size.width} × {size.height} → {rect.width} × {rect.height}</span></div>
    <div className="segmented" role="group" aria-label="Aspect ratio">
      {CROP_RATIOS.map(item => <button key={item} aria-pressed={ratio === item} disabled={busy} onClick={() => onRatio(item)}>{RATIO_LABELS[item]}</button>)}
    </div>
    <div className="crop-size">
      <PixelField label="W" name="Width in pixels" value={rect.width} onCommit={width => onSize(width, rect.height, "width")} />
      <span>×</span>
      <PixelField label="H" name="Height in pixels" value={rect.height} onCommit={height => onSize(rect.width, height, "height")} />
    </div>
    <div className="proposal-actions">
      <button disabled={busy} onClick={onReset}>Reset</button>
      <button disabled={busy} onClick={onCancel}>Cancel</button>
      <button className="primary" disabled={busy || !validCrop(size, rect)} title="Crop the canvas (Enter)" onClick={onApply}>Crop</button>
    </div>
  </section>;
}
