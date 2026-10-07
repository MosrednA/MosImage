import { describe, expect, it } from "vitest";

import { GENERATION_PRESETS, presetForModel } from "./presets";

describe("generation presets", () => {
  it("uses the distilled FLUX settings for quality", () => {
    expect(GENERATION_PRESETS.quality).toMatchObject({
      modelName: "flux2-klein-4b",
      steps: 4,
      guidance: 1,
    });
  });

  it("maps known model names back to the visible preset", () => {
    expect(presetForModel("z-image-turbo")).toBe("fast");
    expect(presetForModel("flux2-klein-4b")).toBe("quality");
  });
});
