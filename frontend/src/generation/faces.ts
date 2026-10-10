import { FaceBox, FaceCandidate } from "../api";

/** Fix faces redraws up to this many faces at a time, as the engine does. */
export const MAX_FACES = 8;

/** What Fix faces knows of the faces on the canvas: how many it found and how many are picked. */
export interface FacePicks { found: number; picked: number }

/** The faces Fix faces picks by itself: the largest real ones, centred inside the selection if there is one. */
export function defaultPicks(faces: readonly FaceCandidate[], inside: readonly boolean[] | null): boolean[] {
  let room = MAX_FACES;
  // Candidates come largest first.
  return faces.map((face, index) => {
    const pick = face.suggested && (inside?.[index] ?? true) && room > 0;
    if (pick) room -= 1;
    return pick;
  });
}

/** The picks after the user's clicks, each of which flips one face. */
export function pickedFaces(defaults: readonly boolean[], flipped: ReadonlySet<number>): boolean[] {
  return defaults.map((pick, index) => pick !== flipped.has(index));
}

/** Flips one face; null when that would pick more faces than Fix faces redraws. */
export function flipFace(picked: readonly boolean[], flipped: ReadonlySet<number>, index: number): Set<number> | null {
  if (!picked[index] && picked.filter(Boolean).length >= MAX_FACES) return null;
  const next = new Set(flipped);
  if (!next.delete(index)) next.add(index);
  return next;
}

/** Whether each face's centre lies in the selection, read the way the engine reads it. */
export function centresInside(faces: readonly FaceCandidate[], alphaAt: (x: number, y: number) => number): boolean[] {
  return faces.map(({ box: [left, top, right, bottom] }) => alphaAt(Math.floor((left + right) / 2), Math.floor((top + bottom) / 2)) > 127);
}

export function pickedBoxes(faces: readonly FaceCandidate[], picked: readonly boolean[]): FaceBox[] {
  return faces.filter((_, index) => picked[index]).map(face => face.box);
}
