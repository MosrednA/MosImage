import { MutableRefObject, useEffect, useRef, useState } from "react";
import { DocumentRecord, GenerationRecord, fileUrl } from "../api";
import { ImageSize } from "../canvas/coordinates";
import { errorMessage } from "../errors";
import { LayerResult } from "../generation/submission";
import { FlatLayer, decodeImage, editChangeMask, flatten, loadImage, shownMask } from "../layers/bitmaps";
import { generationPlacement } from "../layers/geometry";
import { isShown } from "../layers/stack";
import { editedLayer } from "./usePicture";

interface Options {
  result: LayerResult | null;
  chosen: GenerationRecord | undefined;
  source: File | null;
  size: ImageSize | null;
  documentRef: MutableRefObject<DocumentRecord | null>;
  picture: () => Promise<File>;
  onError: (message: string) => void;
}

/** Previews a full generation or layer edit exactly as Apply will place it. */
export function useResultPreview({ result, chosen, source, size, documentRef, picture, onError }: Options): string | null {
  const [url, setUrl] = useState<string | null>(null);
  const latest = useRef({ picture, onError });
  latest.current = { picture, onError };

  useEffect(() => {
    setUrl(null);
    if (!result || (result.kind !== "generation" && result.kind !== "layer") || !chosen || !source || !size) return;
    let disposed = false;
    let created: string | null = null;
    void (async () => {
      const image = await loadImage(fileUrl(chosen.output_image_url));
      let blob: Blob;
      if (result.kind === "generation") {
        const background = await decodeImage(await latest.current.picture());
        const dimensions = { width: image.naturalWidth, height: image.naturalHeight };
        const mask = result.diffMask ? editChangeMask(background, [image], size) : null;
        blob = await flatten(background, size, [{ ...dimensions, ...generationPlacement(dimensions, size), image, mask, opacity: 1 }]);
      } else {
        // The edited copy shows where the layer is; the original is hidden, as after Apply.
        const parts = await Promise.all((documentRef.current?.layers ?? []).map(async (layer): Promise<FlatLayer | null> => {
          const mask = await shownMask(layer);
          if (layer.id === result.layerId) return { ...layer, image: await editedLayer(image, layer), mask, opacity: layer.opacity };
          return isShown(layer) ? { ...layer, image: await loadImage(fileUrl(layer.image_url)), mask } : null;
        }));
        blob = await flatten(await decodeImage(source), size, parts.filter((part): part is FlatLayer => part !== null));
      }
      if (!disposed) { created = URL.createObjectURL(blob); setUrl(created); }
    })().catch(err => { if (!disposed) latest.current.onError(errorMessage(err)); });
    return () => { disposed = true; if (created) URL.revokeObjectURL(created); };
  }, [result, chosen, source, size]);

  return url;
}
