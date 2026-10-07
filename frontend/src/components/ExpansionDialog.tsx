import { Expansion } from "../api";
import { ImageSize } from "../canvas/coordinates";
import { expansionForRatio, expansionSize, validExpansion } from "../canvas/expansion";
import { ControlDialog } from "./ControlDialog";

interface Props { size: ImageSize; expansion: Expansion; onChange: (expansion: Expansion) => void; disabled: boolean; onClose: () => void }

const RATIOS: [string, number][] = [["Wide 16:9", 16 / 9], ["Square", 1], ["Portrait 4:5", .8]];
const SIDES = [["left", "Left"], ["top", "Top"], ["right", "Right"], ["bottom", "Bottom"]] as const;

export function ExpansionDialog({ size, expansion, onChange, disabled, onClose }: Props) {
  const result = expansionSize(size, expansion);
  return <ControlDialog title="Canvas size" description="Choose a format or add space on any side. You can also drag the canvas edges." onClose={onClose}>
    <fieldset disabled={disabled} className="expansion-options">
      <div className="expansion-presets">
        {RATIOS.map(([label, ratio]) => <button key={label} onClick={() => onChange(expansionForRatio(size, ratio))}>{label}</button>)}
        <button onClick={() => onChange({ left: 128, top: 128, right: 128, bottom: 128 })}>All sides +128</button>
      </div>
      <div className="margin-fields">
        {SIDES.map(([side, label]) => <label key={side}>{label}<input type="number" min="0" max="2048" step="32" value={expansion[side]} onChange={event => onChange({ ...expansion, [side]: Number(event.target.value) })} /></label>)}
      </div>
      <div className="canvas-size-result">{result.width} × {result.height} px <small>Original pixels stay intact.</small></div>
      {!validExpansion(size, expansion) && <p className="field-error">Add space and keep the canvas within 4096 × 4096 pixels.</p>}
    </fieldset>
  </ControlDialog>;
}
