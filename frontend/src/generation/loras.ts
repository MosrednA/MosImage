import { LoraChoice, LoraFile, LoraLibrary, MAX_LORAS } from "../api";
import { GENERATION_PRESETS, presetForModel } from "./presets";

/** A LoRA you switched on, with the models its file fits. */
export interface ChosenLora extends LoraChoice { models: string[] }

/** The chosen LoRAs whose files are still usable, with the models they fit now; the same array when nothing changed. */
export function currentLoras(chosen: ChosenLora[], library: LoraLibrary): ChosenLora[] {
  const files = new Map(library.loras.filter(file => !file.problem).map(file => [file.name, file]));
  const kept = chosen.flatMap(lora => { const file = files.get(lora.name); return file ? [{ ...lora, models: file.models }] : []; });
  const same = kept.length === chosen.length && kept.every((lora, index) => lora.models.join() === chosen[index].models.join());
  return same ? chosen : kept;
}

/** Why a LoRA's switch is off limits for this job, or null when it can be switched. */
export function loraNote(file: LoraFile, chosen: ChosenLora[], modelName: string, locked: boolean): string | null {
  if (file.problem) return file.problem;
  if (locked) return "Off while Specialized outpaint runs its own LoRA.";
  if (!file.models.length) return "No model on this device fits it.";
  if (!file.models.includes(modelName)) return `Works with ${GENERATION_PRESETS[presetForModel(file.models[0])].label}.`;
  const active = chosen.filter(lora => lora.models.includes(modelName));
  if (!active.some(lora => lora.name === file.name) && active.length >= MAX_LORAS) return `Up to ${MAX_LORAS} at once.`;
  return null;
}
