import { ModelInfo, RuntimeMode } from "../api";
import { ASPECT_PRESETS, AspectPreset, aspectPresetFor } from "../generation/aspectRatios";
import { GENERATION_PRESETS, GenerationPreset, presetForModel } from "../generation/presets";
import { Operation, taskFor } from "../generation/operation";

export interface EditorSettings {
  preset: GenerationPreset; model_name: string; prompt: string;
  runtime_mode: RuntimeMode; seed: string; steps: number; guidance: number;
  width: number; height: number; strength: number; count: number;
  context_padding: number; scene_context: string;
  consistency_lora: boolean; specialized_outpaint: boolean; detail_crop: boolean;
}
export const defaultSettings: EditorSettings = {
  preset: "quality", model_name: "flux2-klein-9b-nvfp4", prompt: "", runtime_mode: "balanced",
  seed: "", steps: 4, guidance: 1, width: 1024, height: 1024, strength: 1, count: 1,
  context_padding: 320, scene_context: "",
  consistency_lora: false, specialized_outpaint: false, detail_crop: false,
};
/** The settings with a preset's model, steps and guidance. */
export function withPreset(settings: EditorSettings, value: GenerationPreset): EditorSettings {
  const config = GENERATION_PRESETS[value];
  return { ...settings, preset: value, model_name: config.modelName, steps: config.steps, guidance: config.guidance };
}
export function qualityOptions(settings: EditorSettings, operation: Operation, blending: boolean) {
  const specialized_outpaint = operation === "expand" && settings.specialized_outpaint;
  // Specialized outpaint is a Fast LoRA. Enhance follows the preset: the detail enhancer joins on Quality.
  const model_name = specialized_outpaint ? GENERATION_PRESETS.fast.modelName : settings.model_name;
  const consistency_lora = ["edit", "replace", "expand"].includes(operation)
    && model_name === GENERATION_PRESETS.quality.modelName && settings.consistency_lora;
  const detail_crop = operation === "replace" && !blending && settings.detail_crop;
  return { model_name, consistency_lora, specialized_outpaint, detail_crop };
}
export function settingsSummary(settings: EditorSettings, operation: Operation, blending: boolean): string {
  const options = qualityOptions(settings, operation, blending);
  const task = taskFor(operation);
  const quality = GENERATION_PRESETS[presetForModel(options.model_name)].label;
  const count = `${settings.count} ${settings.count === 1 ? "proposal" : "proposals"}`;
  return [quality, count, ...(task === "text-to-image" ? [`${settings.width} × ${settings.height}`] : []),
    ...(options.consistency_lora ? ["Consistency"] : []), ...(options.specialized_outpaint ? ["Specialized"] : []),
    ...(options.detail_crop ? ["Fine detail"] : [])].join(" · ");
}

interface Props {
  settings: EditorSettings; onChange: (settings: EditorSettings) => void; models: ModelInfo[]; operation: Operation; blending: boolean; disabled: boolean;
  /** Apple silicon: CPU and GPU share memory, so every model stays resident and the memory setting has no effect. */
  unifiedMemory?: boolean;
}

export function SettingsControls({ settings, onChange, models, operation, blending, disabled, unifiedMemory = false }: Props) {
  const task = taskFor(operation);
  const options = qualityOptions(settings, operation, blending);
  const showConsistency = ["edit", "replace", "expand"].includes(operation);
  const showDetail = operation === "replace" && !blending;
  const consistencyAvailable = options.model_name === GENERATION_PRESETS.quality.modelName;
  const presetLocked = options.specialized_outpaint;
  function preset(value: GenerationPreset) {
    onChange(withPreset(settings, value));
  }
  return <fieldset className="settings-controls" disabled={disabled}>
    <div className="prompt-options">
      <div className="segmented" aria-label="Quality">{(["quality", "fast"] as const).map(value => <button key={value} disabled={presetLocked} aria-pressed={options.model_name === GENERATION_PRESETS[value].modelName} title={GENERATION_PRESETS[value].description} onClick={() => preset(value)}>{GENERATION_PRESETS[value].label}</button>)}</div>
      {task === "text-to-image" && <select aria-label="Image format" value={aspectPresetFor(settings.width, settings.height) ?? "custom"} onChange={event => { const config = ASPECT_PRESETS[event.target.value as AspectPreset]; if (config) onChange({ ...settings, width: config.width, height: config.height }); }}><option value="square">Square · 1:1</option><option value="landscape">Landscape · 4:3</option><option value="portrait">Portrait · 3:4</option><option value="wide">Wide · 16:9</option><option value="custom" disabled>Custom</option></select>}
      <select aria-label="Number of proposals" value={settings.count} onChange={event => onChange({ ...settings, count: Number(event.target.value) })}>{[1, 2, 3, 4].map(count => <option key={count} value={count}>{count} {count === 1 ? "proposal" : "proposals"}</option>)}</select>
    </div>
      {task === "inpaint" && <div className="blend-controls">
        <label>Scene notes <textarea aria-label="Scene notes" rows={3} maxLength={1500} placeholder="Optional: violet dusk, soft light from the left, flowers in front of the subject…" value={settings.scene_context} onChange={event => onChange({ ...settings, scene_context: event.target.value })} /></label>
        <small>Edge softness and surrounding context are set on the canvas, where you can see their effect.</small>
      </div>}
      {(showConsistency || showDetail || task === "outpaint") && <div className="quality-refinements" aria-label="Optional refinements">
        <small>Optional refinements</small>
        {showConsistency && <label className="quality-switch"><span>Consistency<small>{consistencyAvailable ? "Keep colours and framing closer. Can soften larger changes." : "Available with Quality."}</small></span><input type="checkbox" role="switch" aria-label="Consistency" checked={options.consistency_lora} disabled={!consistencyAvailable} onChange={event => onChange({ ...settings, consistency_lora: event.target.checked })} /></label>}
        {task === "outpaint" && <label className="quality-switch"><span>Specialized outpaint<small>Try an alternative edge fill. Uses Fast while enabled.</small></span><input type="checkbox" role="switch" aria-label="Specialized outpaint" checked={options.specialized_outpaint} onChange={event => onChange({ ...settings, specialized_outpaint: event.target.checked })} /></label>}
        {showDetail && <label className="quality-switch"><span>Fine detail<small>Regenerate selected texture. May invent small details.</small></span><input type="checkbox" role="switch" aria-label="Fine detail" checked={options.detail_crop} onChange={event => onChange({ ...settings, detail_crop: event.target.checked })} /></label>}
      </div>}
      <details className="advanced"><summary>Advanced controls</summary><div className="advanced-content">
        <label>Model<select aria-label="Model" value={options.model_name} disabled={presetLocked} onChange={event => { const name = event.target.value; const value = presetForModel(name); const config = GENERATION_PRESETS[value]; onChange({ ...settings, preset: value, model_name: name, steps: config.steps, guidance: config.guidance }); }}>{models.filter(model => model.capabilities.includes(task)).map(model => <option key={model.name} value={model.name}>{model.display_name}</option>)}</select></label>
        {options.model_name === "flux2-klein-9b-nvfp4" && <p>Native NVFP4 for RTX 50 series. The larger model keeps identities and scenes more faithful; its weights are licensed for non-commercial use only.</p>}
        {options.model_name === "flux2-klein-4b-nvfp4" && <p>Native NVFP4 for RTX 50 series. Smaller and faster than 9B, under a commercial-friendly licence; BF16 remains available for comparison.</p>}
        {options.model_name === "flux2-klein-9b" && <p>The original BF16 weights, on the Apple silicon GPU. The larger model keeps identities and scenes more faithful; its weights are licensed for non-commercial use only.</p>}
        {options.model_name === "flux2-klein-4b" && (unifiedMemory
          ? <p>The original BF16 weights, on the Apple silicon GPU. Smaller and about twice as fast as 9B, under a commercial-friendly licence.</p>
          : <p>The original BF16 weights of 4B, for GPUs without NVFP4 support. Slower and larger than NVFP4.</p>)}
        {!unifiedMemory && <label>Memory usage<select value={settings.runtime_mode} onChange={event => onChange({ ...settings, runtime_mode: event.target.value as RuntimeMode })}><option value="balanced">Balanced</option><option value="safe">Lower VRAM</option><option value="performance">GPU resident</option></select></label>}
        <div className="settings-grid"><label>Steps<input type="number" min="1" max="100" disabled={options.specialized_outpaint} value={options.specialized_outpaint ? 4 : settings.steps} onChange={event => onChange({ ...settings, steps: Number(event.target.value) })} /></label><label>Guidance<input type="number" min="0" max="30" step="0.1" disabled={options.specialized_outpaint} value={options.specialized_outpaint ? 1 : settings.guidance} onChange={event => onChange({ ...settings, guidance: Number(event.target.value) })} /></label></div>
        <label>Seed<input type="number" min="0" max="4294967295" placeholder="Random" value={settings.seed} onChange={event => onChange({ ...settings, seed: event.target.value })} /></label>
        {task === "text-to-image" && <div className="settings-grid"><label>Width<input type="number" min="256" max="2048" step="32" value={settings.width} onChange={event => onChange({ ...settings, width: Number(event.target.value) })} /></label><label>Height<input type="number" min="256" max="2048" step="32" value={settings.height} onChange={event => onChange({ ...settings, height: Number(event.target.value) })} /></label></div>}
        {task === "inpaint" && <label>Edit strength<input type="range" min="0.25" max="1" step="0.05" disabled={options.detail_crop} value={options.detail_crop ? 1 : settings.strength} onChange={event => onChange({ ...settings, strength: Number(event.target.value) })} /></label>}
        <small>Proposals run one at a time. Your selection controls which pixels change.</small>
      </div></details>
  </fieldset>;
}
