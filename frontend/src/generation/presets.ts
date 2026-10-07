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
    modelName: "flux2-klein-4b-nvfp4",
    steps: 4,
    guidance: 1,
  },
  fast: {
    label: "Fast",
    description: "Quick iterations",
    modelName: "z-image-turbo",
    steps: 8,
    guidance: 0,
  },
};

export function presetForModel(modelName: string): GenerationPreset {
  return modelName === GENERATION_PRESETS.fast.modelName ? "fast" : "quality";
}

export function isKleinModel(modelName: string): boolean {
  return modelName === "flux2-klein-4b" || modelName === "flux2-klein-4b-nvfp4";
}
