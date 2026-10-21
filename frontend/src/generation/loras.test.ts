import { describe, expect, it } from "vitest";
import { LoraFile } from "../api";
import { defaultSettings, qualityOptions, settingsSummary } from "../components/EditorSettings";
import { ChosenLora, currentLoras, loraNote } from "./loras";

const QUALITY = ["flux2-klein-9b-nvfp4"];
const FAST = ["flux2-klein-4b-nvfp4", "flux2-klein-4b"];
const file = (name: string, models: string[], problem: string | null = null): LoraFile => ({ name, size_bytes: 1, models, problem });
const chosen = (name: string, models: string[], scale = 1): ChosenLora => ({ name, scale, models });

describe("own LoRAs", () => {
  it("sends only the LoRAs that fit the model the job runs on", () => {
    const settings = { ...defaultSettings, loras: [chosen("Ink", QUALITY, .8), chosen("Film", FAST)] };
    expect(qualityOptions(settings, "edit", false).loras).toEqual([{ name: "Ink", scale: .8 }]);
    expect(qualityOptions({ ...settings, model_name: "flux2-klein-4b" }, "generate", false).loras).toEqual([{ name: "Film", scale: 1 }]);
    // Specialized outpaint runs on Fast with its own LoRA alone.
    expect(qualityOptions({ ...settings, specialized_outpaint: true }, "expand", false).loras).toEqual([]);
    expect(qualityOptions({ ...settings, specialized_outpaint: true }, "generate", false).loras).toHaveLength(1);
  });
  it("names one LoRA in the summary and counts several", () => {
    expect(settingsSummary({ ...defaultSettings, loras: [chosen("Ink", QUALITY)] }, "generate", false)).toMatch(/ · Ink$/);
    const two = { ...defaultSettings, loras: [chosen("Ink", QUALITY), chosen("Grain", QUALITY), chosen("Film", FAST)] };
    expect(settingsSummary(two, "generate", false)).toMatch(/ · 2 LoRAs$/);
  });
  it("drops removed or unusable files and keeps the array when nothing changed", () => {
    const loras = [chosen("Ink", QUALITY, .5), chosen("Gone", QUALITY), chosen("Broken", FAST)];
    const library = { folder: "loras", loras: [file("Ink", FAST), file("Broken", [], "Not a safetensors file.")] };
    expect(currentLoras(loras, library)).toEqual([chosen("Ink", FAST, .5)]);
    const kept = [chosen("Ink", FAST)];
    expect(currentLoras(kept, library)).toBe(kept);
  });
  it("says why a switch is off limits", () => {
    const three = ["A", "B", "C"].map(name => chosen(name, QUALITY));
    expect(loraNote(file("Ink", [], "Made for another model."), [], QUALITY[0], false)).toBe("Made for another model.");
    expect(loraNote(file("Ink", QUALITY), [], QUALITY[0], true)).toMatch("Specialized outpaint");
    expect(loraNote(file("Film", FAST), [], QUALITY[0], false)).toBe("Works with Fast.");
    expect(loraNote(file("Ink", QUALITY), [], FAST[0], false)).toBe("Works with Quality.");
    expect(loraNote(file("D", QUALITY), three, QUALITY[0], false)).toBe("Up to 3 at once.");
    // A chosen one can always be switched off, and LoRAs waiting for the other model do not count.
    expect(loraNote(file("A", QUALITY), three, QUALITY[0], false)).toBeNull();
    expect(loraNote(file("Film", FAST), three, FAST[0], false)).toBeNull();
  });
});
