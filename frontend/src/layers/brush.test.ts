import { describe, expect, it } from "vitest";
import { mixStroke } from "./brush";

/** Mask pixels (white, coverage in alpha) and stroke pixels (grey, opaque where painted). */
function mask(...alphas: number[]): Uint8ClampedArray { return new Uint8ClampedArray(alphas.flatMap(alpha => [255, 255, 255, alpha])); }
function stroke(...levels: number[]): Uint8ClampedArray { return new Uint8ClampedArray(levels.flatMap(level => [level, level, level, level ? 255 : 0])); }
function alphas(pixels: Uint8ClampedArray): number[] { return Array.from({ length: pixels.length / 4 }, (_, index) => pixels[index * 4 + 3]); }

describe("mask brush", () => {
  it("restores and hides by the stroke's strength and leaves unpainted pixels alone", () => {
    const restored = mask(0, 0, 128);
    mixStroke(restored, stroke(255, 0, 255), false, 1);
    expect(alphas(restored)).toEqual([255, 0, 255]);

    const hidden = mask(255, 255, 200);
    mixStroke(hidden, stroke(255, 0, 128), true, 1);
    expect(alphas(hidden)).toEqual([0, 255, 100]);
  });

  it("caps one stroke at its opacity, however often it crosses a pixel", () => {
    const painted = mask(0, 255);
    mixStroke(painted, stroke(255, 255), false, 0.5);
    expect(alphas(painted)).toEqual([128, 255]);
    const hidden = mask(255);
    mixStroke(hidden, stroke(255), true, 0.25);
    expect(alphas(hidden)).toEqual([191]);
  });
});
