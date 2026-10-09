import { MutableRefObject, useRef } from "react";
import { DocumentRecord, LayerRecord, fileUrl } from "../api";
import { ImageSize } from "../canvas/coordinates";
import { fileFromUrl, flatten, loadImage, loadMask, withTransparencyOf } from "../layers/bitmaps";

/** An edited layer image with the original layer's transparency. */
export async function editedLayer(edited: HTMLImageElement, layer: LayerRecord): Promise<HTMLCanvasElement> {
  return withTransparencyOf(edited, await loadImage(fileUrl(layer.image_url)), layer.width, layer.height);
}

/** The picture as you see it: the background with its visible layers, cached between requests. */
export function usePicture(source: File | null, size: ImageSize | null, sourceUrl: string | null, documentRef: MutableRefObject<DocumentRecord | null>) {
  const composite = useRef<{ key: string; file: File } | null>(null);
  const layerFiles = useRef(new Map<string, File>());

  function visibleLayers(): LayerRecord[] {
    return (documentRef.current?.layers ?? []).filter(layer => layer.visible && layer.opacity > 0);
  }
  /** The background with these layers on top, as PNG. */
  async function flattenLayers(visible: LayerRecord[]): Promise<Blob> {
    if (!source || !size) throw new Error("Open an image first.");
    const background = await createImageBitmap(source);
    try {
      const parts = await Promise.all(visible.map(async layer => ({
        ...layer, image: await loadImage(fileUrl(layer.image_url)), mask: layer.mask_url ? await loadMask(fileUrl(layer.mask_url)) : null,
      })));
      return await flatten(background, size, parts);
    } finally { background.close(); }
  }
  /** What the AI works on: the picture as you see it, with its visible layers. */
  async function editImage(): Promise<File> {
    if (!source) throw new Error("Open an image first.");
    const visible = visibleLayers();
    if (!visible.length) return source;
    const key = JSON.stringify([sourceUrl, visible.map(({ image_path, mask_path, x, y, scale_x, scale_y, rotation, opacity }) => [image_path, mask_path, x, y, scale_x, scale_y, rotation, opacity])]);
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
  /** Downloads the picture; layers are flattened onto it as PNG to keep every edge crisp. */
  async function exportPicture(name: string) {
    if (!sourceUrl || !source) return;
    const visible = visibleLayers();
    let href = sourceUrl;
    let extension = source.type === "image/jpeg" ? ".jpg" : source.type === "image/webp" ? ".webp" : ".png";
    if (visible.length && size) {
      href = URL.createObjectURL(await flattenLayers(visible));
      extension = ".png";
    }
    const anchor = window.document.createElement("a"); anchor.href = href;
    anchor.download = name.replace(/[<>:"/\\|?*]+/g, "-").replace(/\.[^.]+$/, "") + extension;
    anchor.click();
    if (href !== sourceUrl) window.setTimeout(() => URL.revokeObjectURL(href), 1000);
  }

  return { visibleLayers, editImage, layerFile, exportPicture };
}
