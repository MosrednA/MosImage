import { PointerEvent, useLayoutEffect, useRef, useState } from "react";
import { ImageSize } from "./coordinates";
import { CropHandle, CropRect, addsSpace, cropBounds, dragCrop } from "./crop";

interface Props {
  /** The picture as you see it, with its visible layers. */
  imageUrl: string;
  size: ImageSize;
  rect: CropRect;
  ratio: number | null;
  onChange: (rect: CropRect) => void;
  disabled: boolean;
}

const HANDLES: readonly Exclude<CropHandle, "move">[] = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

/** Crops the canvas like Photoshop's Crop tool: drag the edges in to cut, out to add transparent space. */
export function CropCanvas({ imageUrl, size, rect, ratio, onChange, disabled }: Props) {
  const stage = useRef<HTMLDivElement>(null);
  const [viewport, setViewport] = useState({ width: 0, height: 0 });
  // The view holds still while dragging, so the handle stays under the pointer.
  const [frozen, setFrozen] = useState<{ bounds: CropRect; zoom: number } | null>(null);
  const drag = useRef<{ handle: CropHandle; x: number; y: number; start: CropRect; zoom: number } | null>(null);
  useLayoutEffect(() => {
    const element = stage.current;
    if (!element) return;
    const observer = new ResizeObserver(() => setViewport({ width: element.clientWidth, height: element.clientHeight }));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const fitBounds = cropBounds(size, rect);
  const fitZoom = Math.max(.02, Math.min((viewport.width - 40) / fitBounds.width, (viewport.height - 40) / fitBounds.height, 1));
  const { bounds, zoom } = frozen ?? { bounds: fitBounds, zoom: fitZoom };
  const place = (x: number, y: number, width: number, height: number) => ({ left: (x - bounds.x) * zoom, top: (y - bounds.y) * zoom, width: width * zoom, height: height * zoom });

  function begin(event: PointerEvent<HTMLDivElement>) {
    if (disabled || event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const handle = ((event.target as HTMLElement).dataset.handle ?? "move") as CropHandle;
    drag.current = { handle, x: event.clientX, y: event.clientY, start: rect, zoom };
    setFrozen({ bounds, zoom });
  }
  function move(event: PointerEvent<HTMLDivElement>) {
    const gesture = drag.current;
    if (!gesture) return;
    onChange(dragCrop(gesture.start, gesture.handle, (event.clientX - gesture.x) / gesture.zoom, (event.clientY - gesture.y) / gesture.zoom, ratio));
  }
  function end() { drag.current = null; setFrozen(null); }

  return <section className="expansion-editor crop-editor">
    <div className="editor-toolbar"><span className="canvas-context-label">Drag the edges in to crop, or out to add empty canvas · Drag inside to move</span></div>
    <div ref={stage} className="expansion-stage">
      <div className="crop-surface" style={{ width: bounds.width * zoom, height: bounds.height * zoom }}>
        <div className="crop-transparent" style={place(rect.x, rect.y, rect.width, rect.height)} />
        <img src={imageUrl} alt="Picture to crop" draggable={false} style={place(0, 0, size.width, size.height)} />
        <div className={`crop-box${disabled ? " disabled" : ""}`} style={place(rect.x, rect.y, rect.width, rect.height)}
          onPointerDown={begin} onPointerMove={move} onPointerUp={end} onPointerCancel={end}>
          <i className="crop-third vertical" style={{ left: "33.333%" }} /><i className="crop-third vertical" style={{ left: "66.667%" }} />
          <i className="crop-third horizontal" style={{ top: "33.333%" }} /><i className="crop-third horizontal" style={{ top: "66.667%" }} />
          {HANDLES.map(handle => <span key={handle} className={`crop-handle handle-${handle}`} data-handle={handle} />)}
        </div>
      </div>
    </div>
    <footer className="canvas-status"><span>{rect.width} × {rect.height} px</span><span>{addsSpace(size, rect) ? "New space stays transparent · Layers move along" : "No AI · Layers move along"}</span></footer>
  </section>;
}
