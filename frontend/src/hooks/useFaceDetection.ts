import { useEffect, useRef, useState } from "react";
import { FaceBox, FaceCandidate, detectFaces } from "../api";
import { ImageSize } from "../canvas/coordinates";
import { errorMessage } from "../errors";
import { FacePicks, MAX_FACES, centresInside, defaultPicks, flipFace, pickedBoxes, pickedFaces } from "../generation/faces";
import { blankCanvas } from "../layers/bitmaps";

interface FaceDetectionOptions {
  /** Fix faces is the operation, so the faces are shown on the canvas. */
  active: boolean;
  /** The picture as you see it, which Fix faces works on. */
  picture: () => Promise<File>;
  /** That picture changes with the source and its layers. */
  source: File | null;
  layers: unknown;
  mask: Blob | null;
  size: ImageSize | null;
  setError: (message: string) => void;
}

/** The faces Fix faces finds in the picture, and which of them the user picked by clicking. */
export function useFaceDetection({ active, picture, source, layers, mask, size, setError }: FaceDetectionOptions) {
  const [found, setFound] = useState<{ file: File; faces: FaceCandidate[] } | null>(null);
  const [finding, setFinding] = useState(false);
  // Whether each face is centred in the selection, for the selection it was read from.
  const [inside, setInside] = useState<{ mask: Blob; values: boolean[] } | null>(null);
  const [flipped, setFlipped] = useState<ReadonlySet<number>>(() => new Set());
  const requested = useRef<File | null>(null);

  async function find(file: File) {
    requested.current = file;
    setFinding(true);
    let faces: FaceCandidate[] | null = null;
    try { faces = await detectFaces(file); } catch (err) { if (requested.current === file) setError(errorMessage(err)); }
    // A newer picture is being searched already.
    if (requested.current !== file) return;
    // Without faces, Fix faces finds them itself; the next change tries again.
    if (faces) setFound({ file, faces }); else requested.current = null;
    setFlipped(new Set()); setFinding(false);
  }

  useEffect(() => {
    if (!active) return;
    picture().then(file => { if (requested.current !== file) void find(file); }).catch(err => setError(errorMessage(err)));
  }, [active, source, layers]);

  useEffect(() => {
    // A new selection starts from the faces Fix faces would pick in it.
    setFlipped(new Set());
    if (!mask || !found || !size) { setInside(null); return; }
    let alive = true;
    createImageBitmap(mask).then(bitmap => {
      const canvas = blankCanvas(size.width, size.height);
      const context = canvas.getContext("2d", { willReadFrequently: true })!;
      context.drawImage(bitmap, 0, 0, size.width, size.height);
      bitmap.close();
      const values = centresInside(found.faces, (x, y) => context.getImageData(Math.min(x, size.width - 1), Math.min(y, size.height - 1), 1, 1).data[3]);
      if (alive) setInside({ mask, values });
    }).catch(err => { if (alive) setError(errorMessage(err)); });
    return () => { alive = false; };
  }, [mask, found, size]);

  const ready = !!found && !finding && (!mask || inside?.mask === mask);
  const faces = ready ? found!.faces : [];
  const picked = ready ? pickedFaces(defaultPicks(faces, mask ? inside!.values : null), flipped) : [];
  const picks: FacePicks | null = ready ? { found: faces.length, picked: picked.filter(Boolean).length } : null;

  return {
    finding, faces, picked, picks,
    toggle(index: number) {
      const next = flipFace(picked, flipped, index);
      if (next) setFlipped(next);
      else setError(`Fix faces redraws up to ${MAX_FACES} faces at a time. Leave one out first.`);
    },
    /** The picked faces when they were found in this picture; otherwise the engine finds the faces itself. */
    boxesFor(file: File): FaceBox[] | null {
      return ready && found?.file === file ? pickedBoxes(faces, picked) : null;
    },
  };
}
