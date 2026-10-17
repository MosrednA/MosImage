import { MutableRefObject, useEffect, useRef, useState } from "react";
import { DocumentRecord, LayerRecord, fileUrl } from "../api";
import { ImageSize } from "../canvas/coordinates";
import { errorMessage } from "../errors";
import { FlatLayer, decodeImage, editedLayerImage, fileFromUrl, flatten, loadImage, shownMask } from "../layers/bitmaps";
import { fillsCanvas, isShown } from "../layers/stack";
import { saveFile } from "../saveFile";

/** An edited layer image following the AI's updated silhouette, to show through the layer's own mask. */
export async function editedLayer(edited: HTMLImageElement, layer: LayerRecord): Promise<HTMLCanvasElement> {
  const [image, mask] = await Promise.all([loadImage(fileUrl(layer.image_url)), shownMask(layer)]);
  return editedLayerImage(edited, image, mask, layer.width, layer.height);
}

/** Layers with their images and masks loaded, ready to draw. */
export function flatLayers(layers: LayerRecord[]): Promise<FlatLayer[]> {
  return Promise.all(layers.map(async layer => ({ ...layer, image: await loadImage(fileUrl(layer.image_url)), mask: await shownMask(layer) })));
}

/** `background` with the layers that show on top, as PNG. */
export async function flattenVisible(background: CanvasImageSource, size: ImageSize, layers: LayerRecord[]): Promise<Blob> {
  return flatten(background, size, await flatLayers(layers.filter(isShown)));
}

/**
 * The picture as you see it, cached between requests. `source` is the empty
 * canvas the layers are drawn on; the picture itself is in its layers.
 */
export function usePicture(source: File | null, size: ImageSize | null, sourceUrl: string | null, documentRef: MutableRefObject<DocumentRecord | null>) {
  const composite = useRef<{ key: string; file: File } | null>(null);
  const layerFiles = useRef(new Map<string, File>());

  function visibleLayers(): LayerRecord[] {
    return (documentRef.current?.layers ?? []).filter(isShown);
  }
  /** These layers on the canvas, as PNG. */
  async function flattenLayers(visible: LayerRecord[]): Promise<Blob> {
    if (!source || !size) throw new Error("Open an image first.");
    return flattenVisible(await decodeImage(source), size, visible);
  }
  /** What the AI works on: the picture as you see it, with its visible layers. */
  async function editImage(): Promise<File> {
    if (!source) throw new Error("Open an image first.");
    const visible = visibleLayers();
    if (!visible.length) return source;
    const key = JSON.stringify([sourceUrl, visible.map(({ image_path, mask_path, mask_enabled, x, y, scale_x, scale_y, rotation, opacity }) => [image_path, mask_path, mask_enabled, x, y, scale_x, scale_y, rotation, opacity])]);
    if (composite.current?.key !== key) composite.current = { key, file: new File([await flattenLayers(visible)], "picture.png", { type: "image/png" }) };
    return composite.current.file;
  }
  /** A layer's own image, without its mask or placement. */
  async function layerFile(layer: LayerRecord): Promise<File> {
    let file = layerFiles.current.get(layer.image_path);
    if (!file) {
      file = await fileFromUrl(fileUrl(layer.image_url), layer.name + ".png");
      layerFiles.current.set(layer.image_path, file);
    }
    return file;
  }
  /**
   * Saves the picture where the user picks, flattened as PNG to keep every
   * edge crisp; a lone background goes out as it came in. Returns where, or
   * null if cancelled.
   */
  async function exportPicture(name: string): Promise<string | null> {
    if (!source || !size) return null;
    const visible = visibleLayers();
    const plain = visible.length === 1 && fillsCanvas(visible[0], size) ? await layerFile(visible[0]) : null;
    const image = plain ?? await flattenLayers(visible);
    const extension = image.type === "image/jpeg" ? ".jpg" : image.type === "image/webp" ? ".webp" : ".png";
    return saveFile(image, name.replace(/[<>:"/\\|?*]+/g, "-").replace(/\.[^.]+$/, "") + extension);
  }

  return { visibleLayers, editImage, layerFile, exportPicture };
}

/** The picture as you see it to show while `active`, following its layers; null until it is ready. */
export function usePictureUrl(active: boolean, picture: () => Promise<File>, source: File | null, layers: LayerRecord[] | undefined, onError: (message: string) => void): string | null {
  const [url, setUrl] = useState<string | null>(null);
  const latest = useRef({ picture, onError });
  latest.current = { picture, onError };

  useEffect(() => {
    if (!active || !source) { setUrl(null); return; }
    let disposed = false;
    // The previous picture stays until this one is ready.
    latest.current.picture().then(file => { if (!disposed) setUrl(URL.createObjectURL(file)); })
      .catch(err => { if (!disposed) latest.current.onError(errorMessage(err)); });
    return () => { disposed = true; };
  }, [active, source, layers]);
  useEffect(() => () => { if (url) URL.revokeObjectURL(url); }, [url]);

  return url;
}
