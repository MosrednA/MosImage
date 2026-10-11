import { describe, expect, it } from "vitest";

import { GENERATION_PRESETS, presetForModel, startingPreset } from "./presets";

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

  it("starts on the preset of the model the engine warms up", () => {
    const mac = [{ name: "flux2-klein-9b", preset: "quality" as const }, { name: "flux2-klein-4b", preset: "fast" as const }];
    expect(startingPreset(mac, "flux2-klein-4b")).toBe("fast");
    expect(startingPreset(mac, "flux2-klein-9b")).toBe("quality");
    // An older engine, or a warm-up model without a preset, leaves the default.
    expect(startingPreset(mac, undefined)).toBeNull();
    expect(startingPreset([{ name: "flux2-klein-4b", preset: null }], "flux2-klein-4b")).toBeNull();
  });
});
