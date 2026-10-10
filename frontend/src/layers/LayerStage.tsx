import { PointerEvent as ReactPointerEvent, MouseEvent as ReactMouseEvent, useEffect, useRef, useState } from "react";
import { LayerRecord, SelectionPoint, fileUrl } from "../api";
import { ImageSize, Point } from "../canvas/coordinates";
import { CanvasTool } from "../canvas/tools";
import { LayerLook, blankCanvas, copyCanvas, loadImage, loadMask, renderLayer, solidAt, solidMask } from "./bitmaps";
import { BrushTip, applyStroke, stampSegment } from "./brush";
import { HANDLES, Handle, canvasToLayer, containsPoint, corners, cssTransform, handlePoint, layerToCanvas, moved, resized, rotated } from "./geometry";
import { layerAt } from "./stack";

interface Props {
  layers: LayerRecord[];
  active: LayerRecord | null;
  zoom: number;
  size: ImageSize;
  tool: CanvasTool;
  brushSize: number;
  /** Mask brush hardness and opacity, 0–1. */
  brushHardness: number;
  brushOpacity: number;
  /** The layer shown as its mask alone, in grey. */
  maskView: string | null;
  /** Tint what the selected layer's mask hides in red. */
  overlay: boolean;
  /** Lasso and Box hide parts of the layer; Alt restores instead. */
  hide: boolean;
  /** Pointer input goes to the layers instead of the AI selection. */
  interactive: boolean;
  disabled: boolean;
  /** Object clicks on the active layer, in layer pixels. */
  points: SelectionPoint[];
  onSelect: (id: string | null) => void;
  onPreview: (id: string, patch: Partial<LayerRecord>) => void;
  onCancelPreview: () => void;
  onCommit: (id: string, patch: Partial<LayerRecord>, label: string) => void;
  onMask: (id: string, mask: HTMLCanvasElement, label: string) => void;
  onObjectPoint: (point: SelectionPoint) => void;
  onRemovePoint: (index: number) => void;
  onCursor: (clientX: number, clientY: number) => void;
}

type Patch = Partial<LayerRecord>;
type Gesture =
  | { kind: "move"; layer: LayerRecord; from: Point; patch: Patch | null }
  | { kind: "resize"; layer: LayerRecord; handle: Handle; patch: Patch | null }
  | { kind: "rotate"; layer: LayerRecord; from: Point; patch: Patch | null }
  // `base` is the mask before the stroke, `stroke` how strongly each pixel is painted.
  | { kind: "paint"; layer: LayerRecord; erase: boolean; tip: BrushTip; opacity: number; base: HTMLCanvasElement; stroke: HTMLCanvasElement; mask: HTMLCanvasElement; image: HTMLImageElement; last: Point }
  | { kind: "shape"; layer: LayerRecord; rectangle: boolean; hide: boolean; points: Point[] };

interface ActiveAssets { imageUrl: string; image: HTMLImageElement; mask: HTMLCanvasElement | null }

const ROTATE_KNOB_DISTANCE = 28;

/** Image layers above the background, with direct manipulation for the selected one. */
export function LayerStage(props: Props) {
  const { layers, active, zoom, size, tool, interactive, disabled } = props;
  const hitRef = useRef<HTMLDivElement | null>(null);
  const canvases = useRef(new Map<string, HTMLCanvasElement>());
  const assets = useRef<ActiveAssets | null>(null);
  const gesture = useRef<Gesture | null>(null);
  const [shape, setShape] = useState<{ points: Point[]; rectangle: boolean } | null>(null);

  // Keep the selected layer's image and mask at hand so painting starts at once.
  useEffect(() => {
    assets.current = null;
    if (!active) return;
    let alive = true;
    const imageUrl = active.image_url;
    Promise.all([loadImage(fileUrl(imageUrl)), active.mask_url ? loadMask(fileUrl(active.mask_url)) : Promise.resolve(null)])
      .then(([image, mask]) => { if (alive) assets.current = { imageUrl, image, mask }; })
      .catch(() => undefined);
    return () => { alive = false; };
  }, [active?.id, active?.image_url, active?.mask_url]);

  function register(id: string, canvas: HTMLCanvasElement | null) {
    if (canvas) canvases.current.set(id, canvas); else canvases.current.delete(id);
  }

  function canvasPoint(event: { clientX: number; clientY: number }): Point {
    const rect = hitRef.current?.getBoundingClientRect();
    if (!rect || !rect.width || !rect.height) return { x: 0, y: 0 };
    return { x: ((event.clientX - rect.left) / rect.width) * size.width, y: ((event.clientY - rect.top) / rect.height) * size.height };
  }

  function solid(layer: LayerRecord, point: Point): boolean {
    const canvas = canvases.current.get(layer.id);
    const local = canvasToLayer(layer, point);
    return canvas ? solidAt(canvas, local.x, local.y) : true;
  }

  function start(event: ReactPointerEvent<HTMLElement>, next: Gesture) {
    event.preventDefault();
    event.stopPropagation();
    gesture.current = next;
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function beginStage(event: ReactPointerEvent<HTMLDivElement>) {
    if (disabled || event.button !== 0) return;
    const point = canvasPoint(event);
    if (tool === "transform") {
      const target = active && containsPoint(active, point) ? active : layerAt(layers, point, layer => solid(layer, point));
      if (!target) { props.onSelect(null); return; }
      if (target.id !== active?.id) props.onSelect(target.id);
      start(event, { kind: "move", layer: target, from: point, patch: null });
      return;
    }
    if (!active) return;
    const local = canvasToLayer(active, point);
    const inside = local.x >= 0 && local.y >= 0 && local.x < active.width && local.y < active.height;
    if (tool === "object") {
      if (inside) props.onObjectPoint({ x: local.x, y: local.y, label: event.altKey ? 0 : 1 });
      return;
    }
    if (tool === "brush" || tool === "eraser") {
      const loaded = assets.current;
      if (!loaded || loaded.imageUrl !== active.image_url) return;
      const base = loaded.mask ? copyCanvas(loaded.mask) : solidMask(active.width, active.height);
      // Brush size is in canvas pixels; the tip works in the layer's own pixels.
      const radius = Math.max(0.5, props.brushSize / 2 / Math.sqrt(Math.abs(active.scale_x * active.scale_y)));
      const paint: Gesture = {
        kind: "paint", layer: active, erase: (tool === "eraser") !== event.altKey, tip: { radius, hardness: props.brushHardness }, opacity: props.brushOpacity,
        base, stroke: blankCanvas(active.width, active.height), mask: copyCanvas(base), image: loaded.image, last: local,
      };
      start(event, paint);
      paintTo(paint, local);
      return;
    }
    if (tool === "lasso" || tool === "rectangle") {
      const rectangle = tool === "rectangle";
      start(event, { kind: "shape", layer: active, rectangle, hide: props.hide !== event.altKey, points: [point] });
      setShape({ points: [point], rectangle });
    }
  }

  function excludePoint(event: ReactMouseEvent<HTMLDivElement>) {
    if (tool !== "object" || !active) return;
    event.preventDefault();
    const local = canvasToLayer(active, canvasPoint(event));
    if (!disabled && local.x >= 0 && local.y >= 0 && local.x < active.width && local.y < active.height) props.onObjectPoint({ x: local.x, y: local.y, label: 0 });
  }

  function beginHandle(event: ReactPointerEvent<HTMLElement>, handle: Handle) {
    if (disabled || !active || event.button !== 0) return;
    start(event, { kind: "resize", layer: active, handle, patch: null });
  }

  function beginRotate(event: ReactPointerEvent<HTMLElement>) {
    if (disabled || !active || event.button !== 0) return;
    start(event, { kind: "rotate", layer: active, from: canvasPoint(event), patch: null });
  }

  function update(event: ReactPointerEvent<HTMLElement>) {
    props.onCursor(event.clientX, event.clientY);
    const current = gesture.current;
    if (!current) return;
    const point = canvasPoint(event);
    if (current.kind === "move" || current.kind === "resize" || current.kind === "rotate") {
      const next = current.kind === "move" ? moved(current.layer, current.from, point)
        : current.kind === "resize" ? resized(current.layer, current.handle, point, event.shiftKey)
        : rotated(current.layer, current.from, point, event.shiftKey);
      current.patch = { x: next.x, y: next.y, scale_x: next.scale_x, scale_y: next.scale_y, rotation: next.rotation };
      props.onPreview(current.layer.id, current.patch);
    } else if (current.kind === "paint") {
      paintTo(current, canvasToLayer(current.layer, point));
    } else {
      const last = current.points[current.points.length - 1];
      if (current.rectangle) current.points = [current.points[0], point];
      else if (Math.hypot(point.x - last.x, point.y - last.y) * zoom >= 2) current.points.push(point);
      setShape({ points: [...current.points], rectangle: current.rectangle });
    }
  }

  function finish(event: ReactPointerEvent<HTMLElement>) {
    const current = gesture.current;
    if (!current) return;
    gesture.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    const id = current.layer.id;
    if (current.kind === "move" || current.kind === "resize" || current.kind === "rotate") {
      if (!current.patch) { props.onCancelPreview(); return; }
      props.onCommit(id, current.patch, current.kind === "move" ? "Move layer" : current.kind === "resize" ? "Resize layer" : "Rotate layer");
    } else if (current.kind === "paint") {
      if (assets.current?.imageUrl === current.layer.image_url) assets.current.mask = current.mask;
      props.onMask(id, current.mask, current.erase ? "Hide part of layer" : "Restore part of layer");
    } else {
      setShape(null);
      fillShape(current);
    }
  }

  function cancel() {
    const current = gesture.current;
    gesture.current = null;
    setShape(null);
    if (!current) return;
    if (current.kind === "paint") redraw(current.layer.id, current.image, assets.current?.mask ?? null);
    else props.onCancelPreview();
  }

  function paintTo(paint: Extract<Gesture, { kind: "paint" }>, point: Point) {
    const context = paint.stroke.getContext("2d");
    if (!context) return;
    const box = stampSegment(context, paint.last, point, paint.tip);
    applyStroke(paint.mask, paint.base, paint.stroke, box, paint.erase, paint.opacity);
    paint.last = point;
    redraw(paint.layer.id, paint.image, paint.mask);
  }

  function fillShape(current: Extract<Gesture, { kind: "shape" }>) {
    const loaded = assets.current;
    const layer = current.layer;
    if (!loaded || loaded.imageUrl !== layer.image_url) return;
    const [first, last] = [current.points[0], current.points[current.points.length - 1]];
    const outline = current.rectangle ? [first, { x: last.x, y: first.y }, last, { x: first.x, y: last.y }] : current.points;
    if (current.rectangle ? Math.abs(last.x - first.x) < 1 || Math.abs(last.y - first.y) < 1 : outline.length < 3) return;
    const mask = loaded.mask ? copyCanvas(loaded.mask) : solidMask(layer.width, layer.height);
    const context = mask.getContext("2d")!;
    context.globalCompositeOperation = current.hide ? "destination-out" : "source-over";
    context.fillStyle = "white";
    context.beginPath();
    outline.map(point => canvasToLayer(layer, point)).forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
    context.closePath();
    context.fill();
    loaded.mask = mask;
    redraw(layer.id, loaded.image, mask);
    props.onMask(layer.id, mask, current.hide ? "Hide part of layer" : "Restore part of layer");
  }

  /** Redraws a layer from a mask being edited, which shows even if it was switched off. */
  function redraw(id: string, image: HTMLImageElement, mask: HTMLCanvasElement | null) {
    const canvas = canvases.current.get(id);
    if (canvas) renderLayer(canvas, image, mask, lookOf(id));
  }
  function lookOf(id: string): LayerLook {
    return id === props.maskView ? "mask" : props.overlay && id === active?.id ? "overlay" : "layer";
  }

  const handlers = { onPointerMove: update, onPointerUp: finish, onPointerCancel: cancel };
  const transforming = !!active && tool === "transform";
  const frame = active ? corners(active).map(point => `${point.x * zoom},${point.y * zoom}`).join(" ") : "";
  const knob = active ? rotateKnob(active, zoom) : null;

  return <div className="layer-stage" {...handlers}>
    <div className="layer-stack" aria-hidden="true">
      {layers.map(layer => <LayerView key={layer.id} layer={layer} look={lookOf(layer.id)} zoom={zoom} register={register} />)}
    </div>
    {interactive && <div ref={hitRef} className="layer-hit" data-tool={tool} onPointerDown={beginStage} onContextMenu={excludePoint} />}
    {!interactive && <div ref={hitRef} className="layer-hit inert" />}
    {active && <svg className="layer-frame" width={size.width * zoom} height={size.height * zoom} aria-hidden="true">
      <polygon points={frame} className={transforming ? "" : "quiet"} />
      {transforming && knob && <line x1={knob.base.x} y1={knob.base.y} x2={knob.tip.x} y2={knob.tip.y} />}
      {shape && <polygon className="layer-shape" points={(shape.rectangle && shape.points.length > 1
        ? [shape.points[0], { x: shape.points[1].x, y: shape.points[0].y }, shape.points[1], { x: shape.points[0].x, y: shape.points[1].y }]
        : shape.points).map(point => `${point.x * zoom},${point.y * zoom}`).join(" ")} />}
    </svg>}
    {transforming && active && !disabled && <>
      {HANDLES.map(handle => {
        const point = handlePoint(active, handle);
        return <div key={handle} className="layer-handle" style={{ left: point.x * zoom, top: point.y * zoom, cursor: resizeCursor(handle, active.rotation) }}
          onPointerDown={event => beginHandle(event, handle)} />;
      })}
      {knob && <div className="layer-rotate" title="Rotate · Shift snaps to 15°" style={{ left: knob.tip.x, top: knob.tip.y }} onPointerDown={beginRotate} />}
    </>}
    {active && tool === "object" && props.points.map((point, index) => {
      const at = layerToCanvas(active, point);
      return <button key={`${index}-${point.x}-${point.y}`} type="button" className={`selection-point ${point.label ? "positive" : "negative"}`}
        style={{ left: at.x * zoom, top: at.y * zoom }} disabled={disabled} aria-label={`Remove ${point.label ? "include" : "exclude"} point ${index + 1}`} title="Click to remove this point"
        onPointerDown={event => event.stopPropagation()} onClick={() => props.onRemovePoint(index)}>
        <span aria-hidden="true">{point.label ? "+" : "−"}</span>
      </button>;
    })}
  </div>;
}

function LayerView({ layer, look, zoom, register }: { layer: LayerRecord; look: LayerLook; zoom: number; register: (id: string, canvas: HTMLCanvasElement | null) => void }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    register(layer.id, ref.current);
    return () => register(layer.id, null);
  }, [layer.id]);
  useEffect(() => {
    let alive = true;
    // The grey and red looks show the mask even while it is switched off.
    const useMask = !!layer.mask_url && (layer.mask_enabled || look !== "layer");
    Promise.all([loadImage(fileUrl(layer.image_url)), useMask ? loadMask(fileUrl(layer.mask_url!)) : Promise.resolve(null)])
      .then(([image, mask]) => { if (alive && ref.current) renderLayer(ref.current, image, mask, look); })
      .catch(() => undefined);
    return () => { alive = false; };
  }, [layer.image_url, layer.mask_url, layer.mask_enabled, layer.width, layer.height, look]);
  return <canvas ref={ref} width={layer.width} height={layer.height}
    style={{ width: layer.width, height: layer.height, transform: cssTransform(layer, zoom), opacity: look === "mask" ? 1 : layer.opacity, visibility: layer.visible || look === "mask" ? undefined : "hidden" }} />;
}

/** The rotate knob sits just outside the top edge, at a fixed distance on screen. */
function rotateKnob(layer: LayerRecord, zoom: number): { base: Point; tip: Point } {
  const top = handlePoint(layer, "n");
  const dx = top.x - layer.x, dy = top.y - layer.y, length = Math.hypot(dx, dy) || 1;
  const base = { x: top.x * zoom, y: top.y * zoom };
  return { base, tip: { x: base.x + (dx / length) * ROTATE_KNOB_DISTANCE, y: base.y + (dy / length) * ROTATE_KNOB_DISTANCE } };
}

const HANDLE_ANGLE: Record<Handle, number> = { e: 0, se: 45, s: 90, sw: 135, w: 180, nw: 225, n: 270, ne: 315 };
const CURSORS = ["ew-resize", "nwse-resize", "ns-resize", "nesw-resize"];

function resizeCursor(handle: Handle, rotation: number): string {
  const angle = (((HANDLE_ANGLE[handle] + rotation) % 180) + 180) % 180;
  return CURSORS[Math.round(angle / 45) % 4];
}
