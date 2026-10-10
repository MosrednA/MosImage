import { CSSProperties } from "react";
import { Icon } from "../components/Icon";
import { OPACITY_RANGE, SOFTNESS_RANGE, TONE_RANGE } from "../layers/brush";

/** Softness, tone (100 white, 0 black) and opacity of the brush while painting a layer mask, in percent. */
export interface MaskBrushControls {
  softness: number;
  onSoftness: (softness: number) => void;
  tone: number;
  onTone: (tone: number) => void;
  opacity: number;
  onOpacity: (opacity: number) => void;
}

interface Props extends MaskBrushControls {
  /** The Eraser always paints black, so only the Brush has a tone. */
  erasing: boolean;
  disabled: boolean;
}

export function toneName(tone: number): string {
  return tone === TONE_RANGE.max ? "White" : tone === TONE_RANGE.min ? "Black" : `${tone}% grey`;
}

/**
 * The mask brush in a bar of its own, under the tool options like the
 * selection edges, so it is clear the strokes change the layer's mask and not its pixels.
 */
export function MaskBrushBar(props: Props) {
  const { disabled, tone } = props;
  return <div className="selection-bar mask-brush-bar" role="toolbar" aria-label="Mask brush">
    <span className="mask-brush-title"><Icon name="invert" />Layer mask</span>
    {props.erasing
      ? <span className="mask-brush-note" title="The Eraser always hides; pick the Brush to paint grey or white">Eraser paints black: hides the layer</span>
      : <div className="edge-slider tone-control" style={{ "--tone": `hsl(0 0% ${tone}%)` } as CSSProperties}>
        <span>Tone</span>
        <button type="button" className="tone-swatch" disabled={disabled} title="Swap white and black · X" aria-label="Swap white and black" onClick={() => props.onTone(TONE_RANGE.max - tone)} />
        <input type="range" aria-label="Mask tone" title="The grey the brush paints: white shows the layer, black hides it" min={TONE_RANGE.min} max={TONE_RANGE.max} value={tone} disabled={disabled} onChange={event => props.onTone(Number(event.target.value))} />
        <output>{toneName(tone)}</output>
      </div>}
    <div className="edge-slider" title="Soft edges blend the layer in; hard edges cut it cleanly">
      <span>Softness</span>
      <input aria-label="Mask brush softness" type="range" min={SOFTNESS_RANGE.min} max={SOFTNESS_RANGE.max} value={props.softness} disabled={disabled} onChange={event => props.onSoftness(Number(event.target.value))} />
      <output>{props.softness}%</output>
    </div>
    <div className="edge-slider" title="How far one stroke moves the mask toward its tone">
      <span>Opacity</span>
      <input aria-label="Mask brush opacity" type="range" min={OPACITY_RANGE.min} max={OPACITY_RANGE.max} value={props.opacity} disabled={disabled} onChange={event => props.onOpacity(Number(event.target.value))} />
      <output>{props.opacity}%</output>
    </div>
    <span className="mask-brush-legend">White shows · black hides · X swaps</span>
  </div>;
}
