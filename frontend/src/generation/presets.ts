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

/** Both 4B variants count as Fast; 9B is Quality. */
export function presetForModel(modelName: string): GenerationPreset {
  return modelName.startsWith("flux2-klein-4b") ? "fast" : "quality";
}
