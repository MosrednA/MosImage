import { MutableRefObject, useEffect, useRef, useState } from "react";
import { DocumentRecord, GenerationRecord, RevisionRecord, fileUrl } from "../api";
import { ImageSize } from "../canvas/coordinates";
import { errorMessage } from "../errors";
import { LayerResult } from "../generation/submission";
import { FlatLayer, flatten, loadImage, shownMask } from "../layers/bitmaps";
import { generationPlacement } from "../layers/geometry";
import { editedLayer, flattenVisible, isShown } from "./usePicture";

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
        const background = await createImageBitmap(await latest.current.picture());
        try {
          const dimensions = { width: image.naturalWidth, height: image.naturalHeight };
          blob = await flatten(background, size, [{ ...dimensions, ...generationPlacement(dimensions, size), image, mask: null, opacity: 1 }]);
        } finally { background.close(); }
      } else {
        // The edited copy shows where the layer is; the original is hidden, as after Apply.
        const parts = await Promise.all((documentRef.current?.layers ?? []).map(async (layer): Promise<FlatLayer | null> => {
          const mask = await shownMask(layer);
          if (layer.id === result.layerId) return { ...layer, image: await editedLayer(image, layer), mask, opacity: layer.opacity };
          return layer.visible && layer.opacity > 0 ? { ...layer, image: await loadImage(fileUrl(layer.image_url)), mask } : null;
        }));
        const background = await createImageBitmap(source);
        try { blob = await flatten(background, size, parts.filter((part): part is FlatLayer => part !== null)); } finally { background.close(); }
      }
      if (!disposed) { created = URL.createObjectURL(blob); setUrl(created); }
    })().catch(err => { if (!disposed) latest.current.onError(errorMessage(err)); });
    return () => { disposed = true; if (created) URL.revokeObjectURL(created); };
  }, [result, chosen, source, size]);

  return url;
}

/**
 * Previews an Expand proposal as Apply leaves it: the new background with the
 * layers moved along on top. Without visible layers that is the proposal itself.
 */
export function useExpansionPreview(proposal: RevisionRecord | null, index: number, onError: (message: string) => void): string | null {
  const [url, setUrl] = useState<string | null>(null);
  const latest = useRef(onError);
  latest.current = onError;
  const alternative = proposal?.alternatives[index];
  const layered = !!proposal?.layers.some(isShown);

  useEffect(() => {
    setUrl(null);
    if (!proposal || !alternative || !layered) return;
    let disposed = false;
    let created: string | null = null;
    void (async () => {
      const image = await loadImage(fileUrl(alternative.output_image_url));
      const blob = await flattenVisible(image, { width: image.naturalWidth, height: image.naturalHeight }, proposal.layers);
      if (!disposed) { created = URL.createObjectURL(blob); setUrl(created); }
    })().catch(err => { if (!disposed) latest.current(errorMessage(err)); });
    return () => { disposed = true; if (created) URL.revokeObjectURL(created); };
  }, [proposal, alternative, layered]);

  return alternative && !layered ? fileUrl(alternative.output_image_url) : url;
}
