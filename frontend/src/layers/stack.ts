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

/** Which variant a layer shows, or -1 for an ordinary layer. */
export function variantIndex(layer: LayerRecord): number {
  return layer.variant_paths.length > 1 ? layer.variant_paths.indexOf(layer.image_path) : -1;
}

/** The patch that shows the next (or previous) variant, wrapping around; null without variants. */
export function stepVariant(layer: LayerRecord, step: number): Pick<LayerRecord, "image_path" | "image_url"> | null {
  const count = layer.variant_paths.length;
  if (count < 2) return null;
  const index = ((Math.max(variantIndex(layer), 0) + step) % count + count) % count;
  return { image_path: layer.variant_paths[index], image_url: layer.variant_urls[index] };
}

/**
 * Turns a layer's variants into layers of their own, just above it in order.
 * The original keeps the variant it shows; the others start hidden.
 */
export function splitVariants(layers: LayerRecord[], id: string): { layers: LayerRecord[]; added: number } | null {
  const index = layers.findIndex(layer => layer.id === id);
  const original = layers[index];
  if (!original || original.variant_paths.length < 2) return null;
  const copies = original.variant_paths.flatMap((path, number) => path === original.image_path ? [] : [{
    ...original, id: newLayerId(), name: `${original.name} · ${number + 1}`.slice(0, 80),
    image_path: path, image_url: original.variant_urls[number], variant_paths: [], variant_urls: [], visible: false,
  }]);
  const kept = { ...original, variant_paths: [], variant_urls: [] };
  return { layers: [...layers.slice(0, index), kept, ...copies, ...layers.slice(index + 1)], added: copies.length };
}

/**
 * Adds an AI result: on top of the stack, or just above the layer it blends
 * in, which is then hidden so the blended version replaces it.
 */
export function insertResult(layers: LayerRecord[], result: LayerRecord, replaces: string | null): LayerRecord[] {
  const index = replaces ? layers.findIndex(layer => layer.id === replaces) : -1;
  if (index < 0) return [...layers, result];
  return [...layers.slice(0, index), { ...layers[index], visible: false }, result, ...layers.slice(index + 1)];
}

/** The topmost visible layer under a canvas point; `solid` can reject transparent pixels. */
export function layerAt<T extends LayerRecord>(layers: T[], point: Point, solid: (layer: T) => boolean = () => true): T | null {
  for (let index = layers.length - 1; index >= 0; index -= 1) {
    const layer = layers[index];
    if (layer.visible && layer.opacity > 0 && containsPoint(layer, point) && solid(layer)) return layer;
  }
  return null;
}
