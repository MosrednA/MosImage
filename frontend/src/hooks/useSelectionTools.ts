import { useEffect, useState } from "react";
import { JobResponse, LayerRecord, SelectionPoint, cutoutSubject, selectByText, selectObject } from "../api";
import { JobExtras } from "../generation/submission";
import { useLayers } from "./useLayers";

/** Clicks keep refining one selection up to this many points. */
const MAX_POINTS = 24;

interface Options {
  source: File | null;
  locked: boolean;
  /** The layer whose mask the tools edit, if any; otherwise they select on the picture. */
  activeLayer: LayerRecord | null;
  layers: Pick<ReturnType<typeof useLayers>, "layers" | "activeId" | "settled" | "setMask">;
  perform: (action: () => Promise<void>) => Promise<void>;
  queued: (created: JobResponse, operation?: null, extras?: JobExtras) => void;
  picture: () => Promise<File>;
  layerFile: (layer: LayerRecord) => Promise<File>;
  setError: (message: string) => void;
  /** Clearing the selection also ends a blend. */
  onClear: () => void;
}

/** The selection on the picture and the object points on a masked layer, with the tools that make them. */
export function useSelectionTools({ source, locked, activeLayer, layers, perform, queued, picture, layerFile, setError, onClear }: Options) {
  const [mask, setMask] = useState<Blob | null>(null);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [points, setPoints] = useState<SelectionPoint[]>([]);
  const [layerPoints, setLayerPoints] = useState<SelectionPoint[]>([]);
  const [clearToken, setClearToken] = useState(0);

  useEffect(() => { setLayerPoints([]); }, [layers.activeId]);

  function clear() { setClearToken(value => value + 1); setMask(null); setMaskUrl(null); setPoints([]); onClear(); }
  /** Forgets everything for a new picture. */
  function reset() { setMask(null); setMaskUrl(null); setPoints([]); setLayerPoints([]); }
  /** Shows a ready-made selection on the picture, such as a layer's footprint. */
  function show(url: string) { setPoints([]); setMaskUrl(url); }

  async function cutOutLayer(layerId: string, url: string, label: string) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) throw new Error("Could not load the selection. Click the object again.");
    await layers.setMask(layerId, await response.blob(), label);
  }

  async function objectPoint(point: SelectionPoint) {
    if (!source || locked) return;
    if (activeLayer) { await layerObjectPoint(activeLayer, point); return; }
    if (!point.label && !points.length) { setError("Click the object you want to select first."); return; }
    if (points.length >= MAX_POINTS) { setError("Start a new selection to continue refining."); return; }
    await perform(async () => {
      const next = [...points, point];
      queued(await selectObject(await picture(), next), null); setPoints(next);
    });
  }
  async function removePoint(index: number) {
    if (!source || locked || index < 0 || index >= points.length) return;
    const next = points.filter((_, position) => position !== index);
    // Without an include point, the remaining exclusions select nothing.
    if (!next.some(point => point.label)) { clear(); return; }
    await perform(async () => { queued(await selectObject(await picture(), next), null); setPoints(next); });
  }
  async function layerObjectPoint(layer: LayerRecord, point: SelectionPoint) {
    if (locked) return;
    if (!point.label && !layerPoints.length) { setError("Click the part of the layer you want to keep first."); return; }
    if (layerPoints.length >= MAX_POINTS) { setError("Start again to keep refining: press Esc, then click the object."); return; }
    const clamped = { ...point, x: Math.min(point.x, layer.width - 1), y: Math.min(point.y, layer.height - 1) };
    await perform(async () => {
      const next = [...layerPoints, clamped];
      await layers.settled();
      queued(await selectObject(await layerFile(layer), next), null, { layerId: layer.id }); setLayerPoints(next);
    });
  }
  async function removeLayerPoint(index: number) {
    const layer = activeLayer;
    if (!layer || locked || index < 0 || index >= layerPoints.length) return;
    const next = layerPoints.filter((_, position) => position !== index);
    // Without a keep point there is nothing to cut out; leave the layer as it is.
    if (!next.some(point => point.label)) { setLayerPoints([]); return; }
    await perform(async () => { queued(await selectObject(await layerFile(layer), next), null, { layerId: layer.id }); setLayerPoints(next); });
  }
  /** Selects what matches a description; while masking a layer, keeps only that on the layer. */
  async function findByText(text: string) {
    if (!source || locked) return;
    const layer = activeLayer;
    await perform(async () => {
      if (layer) {
        await layers.settled();
        queued(await selectByText(await layerFile(layer), text), null, { layerId: layer.id, maskLabel: `Keep “${text}”` }); setLayerPoints([]);
        return;
      }
      queued(await selectByText(await picture(), text), null); setPoints([]);
    });
  }
  /** Keeps only the main subject of a layer, as its layer mask. */
  async function removeLayerBackground(id: string) {
    const layer = layers.layers.find(item => item.id === id);
    if (!layer || locked) return;
    await perform(async () => {
      await layers.settled();
      queued(await cutoutSubject(await layerFile(layer)), null, { layerId: layer.id, maskLabel: "Remove background" }); setLayerPoints([]);
    });
  }
  /** Selects the main subject; while masking a layer, keeps only the subject on the layer. */
  async function selectSubject() {
    if (!source || locked) return;
    if (activeLayer) { await removeLayerBackground(activeLayer.id); return; }
    await perform(async () => { queued(await cutoutSubject(await picture()), null); setPoints([]); });
  }

  return {
    mask, setMask, maskUrl, setMaskUrl, points, layerPoints, setLayerPoints, clearToken,
    clear, reset, show, cutOutLayer, objectPoint, removePoint, removeLayerPoint, findByText, selectSubject, removeLayerBackground,
  };
}
