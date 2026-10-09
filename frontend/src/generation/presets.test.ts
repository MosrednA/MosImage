import { describe, expect, it } from "vitest";

import { GENERATION_PRESETS, presetForModel } from "./presets";

describe("generation presets", () => {
  it("uses the distilled FLUX.2 Klein settings for both presets", () => {
    expect(GENERATION_PRESETS.quality).toMatchObject({ modelName: "flux2-klein-9b-nvfp4", steps: 4, guidance: 1 });
    expect(GENERATION_PRESETS.fast).toMatchObject({ modelName: "flux2-klein-4b-nvfp4", steps: 4, guidance: 1 });
  });

  it("maps known model names back to the visible preset", () => {
    expect(presetForModel("flux2-klein-9b-nvfp4")).toBe("quality");
    expect(presetForModel("flux2-klein-4b-nvfp4")).toBe("fast");
    expect(presetForModel("flux2-klein-4b")).toBe("fast");
  });
});
