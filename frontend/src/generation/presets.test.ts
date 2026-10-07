import { describe, expect, it } from "vitest";

import { GENERATION_PRESETS, isKleinModel, presetForModel } from "./presets";

describe("generation presets", () => {
  it("uses the distilled FLUX settings for quality", () => {
    expect(GENERATION_PRESETS.quality).toMatchObject({
      modelName: "flux2-klein-4b-nvfp4",
      steps: 4,
      guidance: 1,
    });
  });

  it("maps known model names back to the visible preset", () => {
    expect(presetForModel("z-image-turbo")).toBe("fast");
    expect(presetForModel("flux2-klein-4b")).toBe("quality");
    expect(presetForModel("flux2-klein-4b-nvfp4")).toBe("quality");
    expect(isKleinModel("flux2-klein-4b")).toBe(true);
    expect(isKleinModel("flux2-klein-4b-nvfp4")).toBe(true);
    expect(isKleinModel("z-image-turbo")).toBe(false);
  });
});
