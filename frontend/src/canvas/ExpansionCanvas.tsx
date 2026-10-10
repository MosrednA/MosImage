import { PointerEvent, useLayoutEffect, useRef, useState } from "react";
import { Expansion } from "../api";
import { ImageSize } from "./coordinates";
import { expansionSize } from "./expansion";

interface Props {
  /** The picture as you see it; null while it is being prepared. */
  imageUrl: string | null;
  size: ImageSize;
  expansion: Expansion;
  onChange: (expansion: Expansion) => void;
  previewUrl: string | null;
  previewSize: ImageSize | null;
  disabled: boolean;
  reviewing: boolean;
}

export function ExpansionCanvas({ imageUrl, size, expansion, onChange, previewUrl, previewSize, disabled, reviewing }: Props) {
  const stage = useRef<HTMLDivElement>(null);
  const drag = useRef<{ side: keyof Expansion; x: number; y: number; initial: Expansion; zoom: number } | null>(null);
  const [viewport, setViewport] = useState({ width: 0, height: 0 });
  const [manualZoom, setManualZoom] = useState<number | null>(null);
  useLayoutEffect(() => {
    const element = stage.current;
    if (!element) return;
    const observer = new ResizeObserver(() => setViewport({ width: element.clientWidth, height: element.clientHeight }));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const target = previewSize && previewUrl ? previewSize : expansionSize(size, expansion);
  const fitZoom = Math.max(0.03, Math.min((viewport.width - 80) / target.width, (viewport.height - 60) / target.height, 1));
  const zoom = manualZoom ?? fitZoom;
  function changeZoom(value: number) { setManualZoom(Math.min(2, Math.max(.1, value))); }
  function begin(event: PointerEvent<HTMLButtonElement>, side: keyof Expansion) {
    if (disabled) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { side, x: event.clientX, y: event.clientY, initial: expansion, zoom };
  }
  function move(event: PointerEvent<HTMLButtonElement>) {
    const gesture = drag.current;
    if (!gesture) return;
    const delta = gesture.side === "left" || gesture.side === "right" ? event.clientX - gesture.x : event.clientY - gesture.y;
    const direction = gesture.side === "left" || gesture.side === "top" ? -1 : 1;
    const value = Math.round(gesture.initial[gesture.side] + direction * delta / gesture.zoom);
    onChange({ ...gesture.initial, [gesture.side]: Math.max(0, Math.min(2048, value)) });
  }
  return <section className="expansion-editor">
    <div className="editor-toolbar"><span className="canvas-context-label">{reviewing ? "Review your proposal" : "Drag an edge to make more room"}</span><div className="zoom-controls"><button aria-label="Zoom out" onClick={() => changeZoom(zoom - .1)}>−</button><button className={manualZoom === null ? "selected" : ""} onClick={() => setManualZoom(null)}>Fit</button><button className={manualZoom === 1 ? "selected" : ""} onClick={() => changeZoom(1)}>100%</button><button aria-label="Zoom in" onClick={() => changeZoom(zoom + .1)}>+</button><output>{Math.round(zoom * 100)}%</output></div></div>
    <div ref={stage} className="expansion-stage">
      <div className="expansion-surface" style={{ width: target.width * zoom, height: target.height * zoom }}>
        {previewUrl ? <img src={previewUrl} alt="Expanded image proposal" className="expansion-preview" /> : <>
          {imageUrl && <img src={imageUrl} alt="Original image" style={{ left: expansion.left * zoom, top: expansion.top * zoom, width: size.width * zoom, height: size.height * zoom }} />}
          <span className="expansion-label">New space</span>
          {(["left", "top", "right", "bottom"] as const).map(side => <button key={side} className={`expansion-handle handle-${side}`} aria-label={`Drag ${side} edge`} disabled={disabled} onPointerDown={event => begin(event, side)} onPointerMove={move} onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }} />)}
        </>}
      </div>
    </div>
    <footer className="canvas-status"><span>{target.width} × {target.height} px</span><span>Original stays intact</span></footer>
  </section>;
}
