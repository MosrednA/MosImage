import { LayerRecord } from "../api";
import { Point } from "../canvas/coordinates";
import { containsPoint } from "./geometry";

export const MAX_LAYERS = 32;
const DUPLICATE_OFFSET = 24;

export function newLayerId(): string {
  const random = typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : Math.random().toString(16).slice(2) + Date.now().toString(16);
  return "layer-" + random.replace(/-/g, "").slice(0, 16);
}

/** A readable layer name from a file name. */
export function layerName(filename: string | undefined, fallbackNumber: number): string {
  const base = (filename ?? "").replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim();
  return (base && base.toLowerCase() !== "image" ? base : `Layer ${fallbackNumber}`).slice(0, 80);
}

export function patchLayer<T extends { id: string }>(layers: T[], id: string, patch: Partial<T>): T[] {
  return layers.map(layer => layer.id === id ? { ...layer, ...patch } : layer);
}

/** Moves a layer to a stack position, bottom first. */
export function moveLayer<T extends { id: string }>(layers: T[], id: string, index: number): T[] {
  const from = layers.findIndex(layer => layer.id === id);
  if (from < 0) return layers;
  const next = layers.filter(layer => layer.id !== id);
  next.splice(Math.min(Math.max(index, 0), next.length), 0, layers[from]);
  return next;
}

/** A copy just above the original, slightly offset so both stay visible. */
export function duplicateLayer(layers: LayerRecord[], id: string): { layers: LayerRecord[]; copy: LayerRecord } | null {
  const index = layers.findIndex(layer => layer.id === id);
  if (index < 0) return null;
  const original = layers[index];
  const copy = { ...original, id: newLayerId(), name: `${original.name} copy`.slice(0, 80), x: original.x + DUPLICATE_OFFSET, y: original.y + DUPLICATE_OFFSET };
  return { layers: [...layers.slice(0, index + 1), copy, ...layers.slice(index + 1)], copy };
}

/** The topmost visible layer under a canvas point; `solid` can reject transparent pixels. */
export function layerAt<T extends LayerRecord>(layers: T[], point: Point, solid: (layer: T) => boolean = () => true): T | null {
  for (let index = layers.length - 1; index >= 0; index -= 1) {
    const layer = layers[index];
    if (layer.visible && layer.opacity > 0 && containsPoint(layer, point) && solid(layer)) return layer;
  }
  return null;
}
