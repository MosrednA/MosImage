import { ModelInfo } from "../api";

export type GenerationPreset = "quality" | "fast";

export interface GenerationPresetConfig {
  label: string;
  description: string;
  modelName: string;
  steps: number;
  guidance: number;
}

export const GENERATION_PRESETS: Record<GenerationPreset, GenerationPresetConfig> = {
  quality: {
    label: "Quality",
    description: "Best prompts and edits",
    modelName: "flux2-klein-9b-nvfp4",
    steps: 4,
    guidance: 1,
  },
  fast: {
    label: "Fast",
    description: "Quicker, lighter model",
    modelName: "flux2-klein-4b-nvfp4",
    steps: 4,
    guidance: 1,
  },
};

/**
 * Points each preset at the model the engine runs for it: NVFP4 on an RTX 50
 * series card, the BF16 weights on Apple silicon. Engines that do not say keep the defaults.
 */
export function applyPresetModels(models: Pick<ModelInfo, "name" | "preset">[]): void {
  for (const preset of ["quality", "fast"] as const) {
    const model = models.find(candidate => candidate.preset === preset);
    if (model) GENERATION_PRESETS[preset].modelName = model.name;
  }
}

/** The editor starts on the preset whose model the engine warms up, which suits the device. */
export function startingPreset(models: Pick<ModelInfo, "name" | "preset">[], warmUpModel?: string): GenerationPreset | null {
  return models.find(model => model.name === warmUpModel)?.preset ?? null;
}

/** Every 4B variant counts as Fast; 9B is Quality. */
export function presetForModel(modelName: string): GenerationPreset {
  return modelName.startsWith("flux2-klein-4b") ? "fast" : "quality";
}
