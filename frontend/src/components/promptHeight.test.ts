import { describe, expect, it } from "vitest";
import { promptHeight } from "./promptHeight";

describe("promptHeight", () => {
  it("keeps the one-line height when the prompt is not focused", () => {
    expect(promptHeight(170, 2, 40, false)).toBe(40);
  });

  it("grows to fit longer text while focused", () => {
    expect(promptHeight(170, 2, 40, true)).toBe(172);
  });

  it("never shrinks below the one-line height", () => {
    expect(promptHeight(38, 2, 40, true)).toBe(40);
  });

  it("rounds fractional content heights up so the last line is not clipped", () => {
    expect(promptHeight(103.4, 2, 40, true)).toBe(106);
  });
});
