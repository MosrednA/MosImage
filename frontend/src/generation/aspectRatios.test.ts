import { describe, expect, it } from "vitest";
import { ASPECT_PRESETS, aspectPresetFor } from "./aspectRatios";

describe("aspect ratio presets", () => {
  it("uses model-friendly dimensions divisible by 16", () => {
    for (const preset of Object.values(ASPECT_PRESETS)) {
      expect(preset.width % 16).toBe(0);
      expect(preset.height % 16).toBe(0);
    }
  });

  it("recognizes exact presets without guessing custom sizes", () => {
    expect(aspectPresetFor(1344, 768)).toBe("wide");
    expect(aspectPresetFor(1000, 1000)).toBeNull();
  });
});
