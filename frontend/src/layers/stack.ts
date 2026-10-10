import { LayerRecord } from "../api";
import { ImageBox } from "../canvas/context";
import { ImageSize, Point } from "../canvas/coordinates";
import { Placement, containsPoint, corners } from "./geometry";

export const MAX_LAYERS = 40;
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
  const copy = { ...original, id: newLayerId(), name: `${original.name} copy`.slice(0, 80), x: original.x + DUPLICATE_OFFSET, y: original.y + DUPLICATE_OFFSET, original_id: null };
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
    image_path: path, image_url: original.variant_urls[number], variant_paths: [], variant_urls: [], visible: false, original_id: null,
  }]);
  const kept = { ...original, variant_paths: [], variant_urls: [] };
  return { layers: [...layers.slice(0, index), kept, ...copies, ...layers.slice(index + 1)], added: copies.length };
}

/**
 * Adds an AI result: on top of the stack, or just above the layer it blends
 * in, which is then hidden as its original so the blended version replaces it.
 */
export function insertResult(layers: LayerRecord[], result: LayerRecord, replaces: string | null): LayerRecord[] {
  const index = replaces ? layers.findIndex(layer => layer.id === replaces) : -1;
  if (index < 0) return [...layers, result];
  return [...layers.slice(0, index), { ...layers[index], visible: false }, { ...result, original_id: replaces }, ...layers.slice(index + 1)];
}

/** Whether a layer shows in the picture. */
export function isShown(layer: Pick<LayerRecord, "visible" | "opacity">): boolean {
  return layer.visible && layer.opacity > 0;
}

/** Whether a layer covers the canvas exactly and plainly, as a background does. */
export function fillsCanvas(layer: LayerRecord, size: ImageSize): boolean {
  return layer.visible && layer.opacity === 1 && !(layer.mask_path && layer.mask_enabled) && layer.width === size.width && layer.height === size.height
    && layer.x === size.width / 2 && layer.y === size.height / 2 && layer.scale_x === 1 && layer.scale_y === 1 && layer.rotation === 0;
}

/** The layer that replaced this one, when it is kept as an original. */
export function replacementOf(layers: LayerRecord[], id: string): LayerRecord | null {
  return layers.find(layer => layer.original_id === id) ?? null;
}

/** Forgets originals that are no longer in the stack. */
function withoutLostOriginals(layers: LayerRecord[]): LayerRecord[] {
  const ids = new Set(layers.map(layer => layer.id));
  return layers.map(layer => layer.original_id && !ids.has(layer.original_id) ? { ...layer, original_id: null } : layer);
}

/** Deletes layers; the originals that deleted AI layers replaced show again, so deleting an edit undoes it. */
export function removeLayers(layers: LayerRecord[], ids: string[]): LayerRecord[] {
  const gone = new Set(ids);
  const restored = new Set(layers.flatMap(layer => gone.has(layer.id) && layer.original_id && !gone.has(layer.original_id) ? [layer.original_id] : []));
  return withoutLostOriginals(layers.filter(layer => !gone.has(layer.id)).map(layer => restored.has(layer.id) ? { ...layer, visible: true } : layer));
}

/** The layers from one to the other, as Shift-click selects them. */
export function rangeIds(layers: LayerRecord[], from: string, to: string): string[] {
  const start = layers.findIndex(layer => layer.id === from);
  const end = layers.findIndex(layer => layer.id === to);
  if (end < 0) return [];
  if (start < 0) return [to];
  return layers.slice(Math.min(start, end), Math.max(start, end) + 1).map(layer => layer.id);
}

/**
 * The merges of Photoshop's Layer menu: Merge Down onto the next visible
 * layer below, the selected layers, every visible layer, or Flatten Image.
 */
export type MergeKind = "down" | "selected" | "visible" | "flatten";

export interface MergePlan {
  kind: MergeKind;
  /** What the merged layer is drawn from, bottom first. */
  parts: LayerRecord[];
  /** The layer whose place in the stack the merged one takes. */
  at: string;
  name: string;
  /** Every layer that goes: the parts, the hidden originals they leave behind and, with Flatten, the hidden layers too. */
  removed: string[];
}

/** What a merge would do, or null when there is nothing to merge. `ids` are the selected layers; Merge Down uses the first. */
export function mergePlan(layers: LayerRecord[], kind: MergeKind, ids: string[], size: ImageSize): MergePlan | null {
  let parts: LayerRecord[];
  let at: LayerRecord;
  if (kind === "down") {
    const index = layers.findIndex(layer => layer.id === ids[0]);
    const below = layers.slice(0, Math.max(index, 0)).filter(isShown).pop();
    if (index < 0 || !isShown(layers[index]) || !below) return null;
    parts = [below, layers[index]];
    at = below;
  } else {
    parts = layers.filter(layer => isShown(layer) && (kind !== "selected" || ids.includes(layer.id)));
    if (!parts.length || (kind === "flatten" ? layers.length === 1 && fillsCanvas(layers[0], size) : parts.length < 2)) return null;
    // Merged selected layers take the top one's place, as in Photoshop; merged visible ones the bottom one's.
    at = kind === "selected" ? parts[parts.length - 1] : parts[0];
  }
  if (kind === "flatten") return { kind, parts, at: at.id, name: "Background", removed: layers.map(layer => layer.id) };
  const removed = new Set(parts.map(layer => layer.id));
  // Hidden originals of merged layers have nothing left to go back to.
  for (let grew = true; grew;) {
    grew = false;
    for (const layer of layers) {
      if (layer.visible || removed.has(layer.id) || !layers.some(other => removed.has(other.id) && other.original_id === layer.id)) continue;
      removed.add(layer.id);
      grew = true;
    }
  }
  return { kind, parts, at: at.id, name: at.name, removed: layers.filter(layer => removed.has(layer.id)).map(layer => layer.id) };
}

/** The stack with a plan's layers replaced by the merged one. */
export function withMerged(layers: LayerRecord[], plan: MergePlan, merged: LayerRecord): LayerRecord[] {
  const removed = new Set(plan.removed);
  return withoutLostOriginals(layers.flatMap(layer => layer.id === plan.at ? [merged] : removed.has(layer.id) ? [] : [layer]));
}

/**
 * The canvas area a merged layer keeps, in whole pixels: everything the parts
 * cover within half a canvas around it, or the canvas itself for Flatten.
 */
export function mergeBounds(parts: Placement[], size: ImageSize, flatten: boolean): ImageBox {
  if (flatten || !parts.length) return { x: 0, y: 0, width: size.width, height: size.height };
  const points = parts.flatMap(corners);
  // A hair of tolerance keeps exact edges from gaining a pixel to rounding.
  const left = Math.max(Math.floor(Math.min(...points.map(point => point.x)) + 1e-6), -Math.round(size.width / 2));
  const top = Math.max(Math.floor(Math.min(...points.map(point => point.y)) + 1e-6), -Math.round(size.height / 2));
  const right = Math.min(Math.ceil(Math.max(...points.map(point => point.x)) - 1e-6), Math.round(size.width * 1.5));
  const bottom = Math.min(Math.ceil(Math.max(...points.map(point => point.y)) - 1e-6), Math.round(size.height * 1.5));
  return { x: left, y: top, width: Math.max(1, right - left), height: Math.max(1, bottom - top) };
}

/** The topmost visible layer under a canvas point; `solid` can reject transparent pixels. */
export function layerAt<T extends LayerRecord>(layers: T[], point: Point, solid: (layer: T) => boolean = () => true): T | null {
  for (let index = layers.length - 1; index >= 0; index -= 1) {
    const layer = layers[index];
    if (layer.visible && layer.opacity > 0 && containsPoint(layer, point) && solid(layer)) return layer;
  }
  return null;
}
