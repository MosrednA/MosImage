import { describe, expect, it } from "vitest";
import { FaceCandidate } from "../api";
import { MAX_FACES, centresInside, defaultPicks, flipFace, pickedBoxes, pickedFaces } from "./faces";

const face = (left: number, suggested = true): FaceCandidate => ({ box: [left, 10, left + 20, 30], score: 0.5, suggested });

describe("picking faces", () => {
  it("picks the real faces, at most eight, and only those in the selection", () => {
    expect(defaultPicks([face(0), face(30, false), face(60)], null)).toEqual([true, false, true]);
    expect(defaultPicks([face(0), face(30, false), face(60)], [false, true, true])).toEqual([false, false, true]);
    const crowd = Array.from({ length: 10 }, (_, index) => face(index * 30));
    expect(defaultPicks(crowd, null).filter(Boolean)).toHaveLength(MAX_FACES);
    expect(defaultPicks(crowd, null).slice(MAX_FACES)).toEqual([false, false]);
  });

  it("flips a face per click, but never picks more than Fix faces redraws", () => {
    const defaults = [true, false];
    const flipped = flipFace(pickedFaces(defaults, new Set()), new Set(), 0)!;
    expect(pickedFaces(defaults, flipped)).toEqual([false, false]);
    expect(pickedFaces(defaults, flipFace(pickedFaces(defaults, flipped), flipped, 0)!)).toEqual([true, false]);
    const full = Array.from({ length: MAX_FACES }, () => true).concat(false);
    expect(flipFace(full, new Set(), MAX_FACES)).toBeNull();
    expect(flipFace(full, new Set(), 0)).toEqual(new Set([0]));
  });

  it("reads the selection at each face's centre and sends the picked boxes", () => {
    const faces = [face(0), face(60)];
    expect(centresInside(faces, x => x < 50 ? 255 : 100)).toEqual([true, false]);
    expect(pickedBoxes(faces, [false, true])).toEqual([[60, 10, 80, 30]]);
  });
});
