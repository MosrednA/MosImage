import { describe, expect, it } from "vitest";
import { changeMaskPixels, editedLayerPixels, footprintPixels, luminanceToAlpha, thresholdAlpha, unmaskedLayerPixels } from "./bitmaps";

describe("layer bitmaps", () => {
  it("stores greyscale mask coverage in alpha over white", () => {
    const data = new Uint8ClampedArray([0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255]);
    luminanceToAlpha(data);
    expect(Array.from(data)).toEqual([255, 255, 255, 0, 255, 255, 255, 128, 255, 255, 255, 255]);
  });

  it("grows a soft footprint into a solid one", () => {
    const data = new Uint8ClampedArray([0, 0, 0, 0, 10, 20, 30, 8, 255, 255, 255, 9, 0, 0, 0, 200]);
    thresholdAlpha(data, 8);
    expect(Array.from(data)).toEqual([255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255]);
  });

  it("grows and softens a footprint without canvas filters", () => {
    const width = 30, height = 30;
    const data = new Uint8ClampedArray(width * height * 4);
    for (let y = 10; y < 20; y += 1) for (let x = 10; x < 20; x += 1) data[(y * width + x) * 4 + 3] = 255;
    footprintPixels(data, width, height, 8, 2);
    const grey = (x: number, y: number) => data[(y * width + x) * 4];
    expect(grey(15, 15)).toBe(255);
    // Grown by about the blur radius, then softened at the new edge.
    expect(grey(8, 15)).toBeGreaterThan(128);
    expect(grey(5, 15)).toBeGreaterThan(0);
    expect(grey(5, 15)).toBeLessThan(255);
    expect(grey(0, 0)).toBe(0);
    expect(data[3]).toBe(255);
  });

  it("preserves AI additions outside a layer's silhouette while keeping background cut out", () => {
    const width = 20, height = 20;
    const original = new Uint8ClampedArray(width * height * 4);
    const edited = new Uint8ClampedArray(width * height * 4);
    // Neutral reference background everywhere in edited.
    for (let i = 0; i < width * height; i += 1) {
      edited[i * 4] = edited[i * 4 + 1] = edited[i * 4 + 2] = 232;
      edited[i * 4 + 3] = 255;
    }
    // Original layer: a blue body in y=[10..15], x=[6..14].
    for (let y = 10; y <= 15; y += 1) {
      for (let x = 6; x <= 14; x += 1) {
        const idx = (y * width + x) * 4;
        original[idx] = 30; original[idx + 1] = 80; original[idx + 2] = 200; original[idx + 3] = 255;
        edited[idx] = 30; edited[idx + 1] = 80; edited[idx + 2] = 200;
      }
    }
    // AI added a red hat outside the old silhouette at y=[4..9], x=[7..13], with a light-grey badge at (10, 6).
    for (let y = 4; y <= 9; y += 1) {
      for (let x = 7; x <= 13; x += 1) {
        const idx = (y * width + x) * 4;
        edited[idx] = 200; edited[idx + 1] = 20; edited[idx + 2] = 20;
      }
    }
    const badgeIdx = (6 * width + 10) * 4;
    edited[badgeIdx] = edited[badgeIdx + 1] = edited[badgeIdx + 2] = 232;

    // AI also erased the bottom row y=15 back to background grey.
    for (let x = 6; x <= 14; x += 1) {
      const idx = (15 * width + x) * 4;
      edited[idx] = edited[idx + 1] = edited[idx + 2] = 232;
    }

    editedLayerPixels(edited, original, width, height);
    const alpha = (x: number, y: number) => edited[(y * width + x) * 4 + 3];
    // Exterior background stays transparent.
    expect(alpha(0, 0)).toBe(0);
    // Original body remains opaque.
    expect(alpha(10, 12)).toBe(255);
    // Added hat outside original silhouette is kept opaque, including its interior light-grey badge.
    expect(alpha(10, 5)).toBe(255);
    expect(alpha(10, 6)).toBe(255);
    // Erased row at y=15 becomes transparent.
    expect(alpha(10, 15)).toBe(0);
  });

  /** A layer row by row from `alphaAt`, in `colour`, and what the engine renders of it over its grey. */
  function softLayer(width: number, height: number, colour: number[], alphaAt: (x: number, y: number) => number, shownAs = colour) {
    const original = new Uint8ClampedArray(width * height * 4);
    const edited = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
      const idx = (y * width + x) * 4, a = alphaAt(x, y);
      original.set([...colour, a], idx);
      edited.set([...shownAs.map(value => Math.round(value * a / 255 + 232 * (1 - a / 255))), 255], idx);
    }
    return { original, edited };
  }
  const RAMP = [0, 0, 0, 0, 32, 64, 96, 128, 160, 192, 223];
  const rampAlpha = (x: number) => RAMP[x] ?? 255;

  it("keeps the soft edges of a layer the edit left alone", () => {
    const width = 20, height = 6;
    const { original, edited } = softLayer(width, height, [40, 60, 200], rampAlpha);
    editedLayerPixels(edited, original, width, height);
    for (let x = 0; x < width; x += 1) {
      const idx = (3 * width + x) * 4;
      expect(edited[idx + 3]).toBe(rampAlpha(x));
      if (rampAlpha(x) >= 64) {
        expect(Math.abs(edited[idx] - 40)).toBeLessThanOrEqual(4);
        expect(Math.abs(edited[idx + 2] - 200)).toBeLessThanOrEqual(4);
      }
    }
  });

  it("keeps the softness of edges the edit recoloured", () => {
    const width = 20, height = 6;
    const { original, edited } = softLayer(width, height, [40, 60, 200], rampAlpha, [200, 40, 40]);
    editedLayerPixels(edited, original, width, height);
    for (let x = 0; x < width; x += 1) expect(edited[(3 * width + x) * 4 + 3]).toBe(rampAlpha(x));
    expect(Math.abs(edited[(3 * width + 6) * 4] - 200)).toBeLessThanOrEqual(4);
  });

  it("keeps small gaps in a layer transparent", () => {
    const width = 20, height = 20;
    const gap = (x: number, y: number) => y === 9 && (x === 9 || x === 10);
    const { original, edited } = softLayer(width, height, [90, 60, 30], (x, y) => x >= 3 && x <= 16 && y >= 3 && y <= 16 && !gap(x, y) ? 255 : 0);
    editedLayerPixels(edited, original, width, height);
    expect(edited[(9 * width + 9) * 4 + 3]).toBe(0);
    expect(edited[(9 * width + 10) * 4 + 3]).toBe(0);
    expect(edited[(8 * width + 9) * 4 + 3]).toBe(255);
  });

  it("puts an edit back under the layer's mask without losing what the mask hides", () => {
    const width = 16, height = 4;
    const raw = new Uint8ClampedArray(width * height * 4);
    const mask = new Uint8ClampedArray(width * height * 4);
    const shownAt = (x: number) => x < 4 ? 0 : x < 10 ? 40 * (x - 3) : 255;
    for (let p = 0; p < width * height; p += 1) {
      raw.set([30, 80, 200, 255], p * 4);
      mask.set([255, 255, 255, shownAt(p % width)], p * 4);
    }
    // The AI sees the layer through its mask and changes nothing.
    const { original: seen, edited } = softLayer(width, height, [30, 80, 200], shownAt);
    editedLayerPixels(edited, seen, width, height);
    unmaskedLayerPixels(edited, raw, mask);
    for (let x = 0; x < width; x += 1) {
      const idx = (2 * width + x) * 4;
      expect(edited[idx + 3]).toBe(255);
      expect(Math.abs(edited[idx] - 30)).toBeLessThanOrEqual(4);
      expect(Math.abs(edited[idx + 2] - 200)).toBeLessThanOrEqual(4);
    }
  });

  it("shows an edit through a partial mask as edited and keeps new detail where it hides", () => {
    const edited = new Uint8ClampedArray([10, 20, 30, 200, 0, 0, 0, 0, 100, 100, 100, 64, 200, 0, 0, 255, 232, 232, 232, 0]);
    const raw = new Uint8ClampedArray([1, 2, 3, 255, 50, 60, 70, 255, 1, 2, 3, 255, 1, 2, 3, 255, 1, 2, 3, 255]);
    const mask = new Uint8ClampedArray([255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 128, 255, 255, 255, 0, 255, 255, 255, 128]);
    unmaskedLayerPixels(edited, raw, mask);
    expect(Array.from(edited)).toEqual([
      10, 20, 30, 200, // fully shown: as edited
      50, 60, 70, 255, // hidden and untouched: the original stays
      100, 100, 100, 128, // half shown: shows at the edited coverage
      200, 0, 0, 255, // new detail under a hiding mask stays for when the mask shows it
      232, 232, 232, 0, // erased where the AI saw it: stays erased
    ]);
  });

  it("builds a feathered difference mask for local edits and returns null for global edits", () => {
    const width = 40, height = 40;
    const source = new Uint8ClampedArray(width * height * 4);
    const localEdit = new Uint8ClampedArray(width * height * 4);
    const globalEdit = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < width * height; i += 1) {
      const idx = i * 4;
      source[idx] = localEdit[idx] = 50;
      source[idx + 1] = localEdit[idx + 1] = 100;
      source[idx + 2] = localEdit[idx + 2] = 150;
      source[idx + 3] = localEdit[idx + 3] = globalEdit[idx + 3] = 255;
      globalEdit[idx] = 200; globalEdit[idx + 1] = 210; globalEdit[idx + 2] = 220;
    }
    // Local edit in a 8x8 patch around the centre.
    for (let y = 16; y < 24; y += 1) {
      for (let x = 16; x < 24; x += 1) {
        const idx = (y * width + x) * 4;
        localEdit[idx] = 220; localEdit[idx + 1] = 40; localEdit[idx + 2] = 40;
      }
    }
    const mask = changeMaskPixels(source, [localEdit], width, height);
    expect(mask).not.toBeNull();
    expect(mask![(20 * width + 20) * 4 + 3]).toBe(255);
    expect(mask![3]).toBe(0);

    // Global edit (> 45% changed) and no-op edit (< 0.2% changed) return null.
    expect(changeMaskPixels(source, [globalEdit], width, height)).toBeNull();
    expect(changeMaskPixels(source, [source], width, height)).toBeNull();
  });
});
