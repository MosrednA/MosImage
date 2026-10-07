export type AspectPreset = "square" | "landscape" | "portrait" | "wide";

export interface AspectPresetConfig {
  label: string;
  width: number;
  height: number;
}

export const ASPECT_PRESETS: Record<AspectPreset, AspectPresetConfig> = {
  square: { label: "Square", width: 1024, height: 1024 },
  landscape: { label: "Landscape", width: 1152, height: 864 },
  portrait: { label: "Portrait", width: 864, height: 1152 },
  wide: { label: "Wide", width: 1344, height: 768 },
};

export function aspectPresetFor(width: number, height: number): AspectPreset | null {
  const entry = (Object.entries(ASPECT_PRESETS) as [AspectPreset, AspectPresetConfig][]).find(
    ([, preset]) => preset.width === width && preset.height === height,
  );
  return entry?.[0] ?? null;
}
