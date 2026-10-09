import { ImageTask, ModelInfo, RuntimeMode } from "../api";
import { ASPECT_PRESETS, AspectPreset, aspectPresetFor } from "../generation/aspectRatios";
import { GENERATION_PRESETS, GenerationPreset, presetForModel } from "../generation/presets";

export interface EditorSettings {
  preset: GenerationPreset; model_name: string; prompt: string;
  runtime_mode: RuntimeMode; seed: string; steps: number; guidance: number;
  width: number; height: number; strength: number; count: number;
  context_padding: number; scene_context: string;
}
export const defaultSettings: EditorSettings = {
  preset: "quality", model_name: "flux2-klein-9b-nvfp4", prompt: "", runtime_mode: "balanced",
  seed: "", steps: 4, guidance: 1, width: 1024, height: 1024, strength: 1, count: 1,
  context_padding: 320, scene_context: "",
};
export function settingsSummary(settings: EditorSettings, task: ImageTask): string {
  const quality = GENERATION_PRESETS[presetForModel(settings.model_name)].label;
  const count = `${settings.count} ${settings.count === 1 ? "proposal" : "proposals"}`;
  return [quality, count, ...(task === "text-to-image" ? [`${settings.width} × ${settings.height}`] : [])].join(" · ");
}

interface Props { settings: EditorSettings; onChange: (settings: EditorSettings) => void; models: ModelInfo[]; task: ImageTask; disabled: boolean }

export function SettingsControls({ settings, onChange, models, task, disabled }: Props) {
  function preset(value: GenerationPreset) {
    const config = GENERATION_PRESETS[value];
    onChange({ ...settings, preset: value, model_name: config.modelName, steps: config.steps, guidance: config.guidance });
  }
  return <fieldset className="settings-controls" disabled={disabled}>
    <div className="prompt-options">
      <div className="segmented" aria-label="Quality">{(["quality", "fast"] as const).map(value => <button key={value} aria-pressed={settings.model_name === GENERATION_PRESETS[value].modelName} title={GENERATION_PRESETS[value].description} onClick={() => preset(value)}>{GENERATION_PRESETS[value].label}</button>)}</div>
      {task === "text-to-image" && <select aria-label="Image format" value={aspectPresetFor(settings.width, settings.height) ?? "custom"} onChange={event => { const config = ASPECT_PRESETS[event.target.value as AspectPreset]; if (config) onChange({ ...settings, width: config.width, height: config.height }); }}><option value="square">Square · 1:1</option><option value="landscape">Landscape · 4:3</option><option value="portrait">Portrait · 3:4</option><option value="wide">Wide · 16:9</option><option value="custom" disabled>Custom</option></select>}
      <select aria-label="Number of proposals" value={settings.count} onChange={event => onChange({ ...settings, count: Number(event.target.value) })}>{[1, 2, 3, 4].map(count => <option key={count} value={count}>{count} {count === 1 ? "proposal" : "proposals"}</option>)}</select>
    </div>
      {task === "inpaint" && <div className="blend-controls">
        <label>Scene notes <textarea aria-label="Scene notes" rows={3} maxLength={1500} placeholder="Optional: violet dusk, soft light from the left, flowers in front of the subject…" value={settings.scene_context} onChange={event => onChange({ ...settings, scene_context: event.target.value })} /></label>
        <small>Edge softness and surrounding context are set on the canvas, where you can see their effect.</small>
      </div>}
      <details className="advanced"><summary>Advanced controls</summary><div className="advanced-content">
        <label>Model<select aria-label="Model" value={settings.model_name} onChange={event => { const name = event.target.value; const value = presetForModel(name); const config = GENERATION_PRESETS[value]; onChange({ ...settings, preset: value, model_name: name, steps: config.steps, guidance: config.guidance }); }}>{models.filter(model => model.capabilities.includes(task)).map(model => <option key={model.name} value={model.name}>{model.display_name}</option>)}</select></label>
        {settings.model_name === "flux2-klein-9b-nvfp4" && <p>Native NVFP4 for RTX 50 series. The larger model keeps identities and scenes more faithful; its weights are licensed for non-commercial use only.</p>}
        {settings.model_name === "flux2-klein-4b-nvfp4" && <p>Native NVFP4 for RTX 50 series. Smaller and faster than 9B, under a commercial-friendly licence; BF16 remains available for comparison.</p>}
        {settings.model_name === "flux2-klein-4b" && <p>The original BF16 weights of 4B, for GPUs without NVFP4 support. Slower and larger than NVFP4.</p>}
        <label>Memory usage<select value={settings.runtime_mode} onChange={event => onChange({ ...settings, runtime_mode: event.target.value as RuntimeMode })}><option value="balanced">Balanced</option><option value="safe">Lower VRAM</option><option value="performance">GPU resident</option></select></label>
        <div className="settings-grid"><label>Steps<input type="number" min="1" max="100" value={settings.steps} onChange={event => onChange({ ...settings, steps: Number(event.target.value) })} /></label><label>Guidance<input type="number" min="0" max="30" step="0.1" value={settings.guidance} onChange={event => onChange({ ...settings, guidance: Number(event.target.value) })} /></label></div>
        <label>Seed<input type="number" min="0" max="4294967295" placeholder="Random" value={settings.seed} onChange={event => onChange({ ...settings, seed: event.target.value })} /></label>
        {task === "text-to-image" && <div className="settings-grid"><label>Width<input type="number" min="256" max="2048" step="32" value={settings.width} onChange={event => onChange({ ...settings, width: Number(event.target.value) })} /></label><label>Height<input type="number" min="256" max="2048" step="32" value={settings.height} onChange={event => onChange({ ...settings, height: Number(event.target.value) })} /></label></div>}
        {task === "inpaint" && <label>Edit strength<input type="range" min="0.25" max="1" step="0.05" value={settings.strength} onChange={event => onChange({ ...settings, strength: Number(event.target.value) })} /></label>}
        <small>Proposals run one at a time. Your selection controls which pixels change.</small>
      </div></details>
  </fieldset>;
}
