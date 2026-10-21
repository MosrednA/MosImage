import { useEffect, useState } from "react";
import { LoraLibrary, MAX_LORAS, listLoras } from "../api";
import { ChosenLora, currentLoras, loraNote } from "../generation/loras";

interface Props {
  chosen: ChosenLora[]; onChange: (loras: ChosenLora[]) => void;
  /** The model the job runs on; a LoRA for the other size waits until you switch back. */
  modelName: string;
  /** Specialized outpaint runs its own LoRA alone. */
  locked: boolean;
}

function megabytes(bytes: number): string {
  return `${Math.max(1, Math.round(bytes / 1048576))} MB`;
}

/** Your own LoRAs from the LoRA folder, each with a switch and a strength. */
export function LoraControls({ chosen, onChange, modelName, locked }: Props) {
  const [library, setLibrary] = useState<LoraLibrary | null>(null);
  const [error, setError] = useState<string | null>(null);
  function load() {
    listLoras().then(next => {
      setLibrary(next); setError(null);
      // A file that was removed or rewritten for the other model no longer counts as chosen.
      const current = currentLoras(chosen, next);
      if (current !== chosen) onChange(current);
    }, (reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)));
  }
  useEffect(load, []);
  function choose(name: string, models: string[], on: boolean) {
    const others = chosen.filter(lora => lora.name !== name);
    onChange(on ? [...others, { name, scale: 1, models }] : others);
  }
  function scale(name: string, value: number) {
    onChange(chosen.map(lora => lora.name === name ? { ...lora, scale: value } : lora));
  }
  return <div className="quality-refinements lora-controls" aria-label="Your LoRAs">
    <div className="lora-heading"><small>Your LoRAs</small><button type="button" onClick={load}>Refresh</button></div>
    {error && <small className="lora-problem">{error}</small>}
    {library && (library.loras.length
      ? <small>Up to {MAX_LORAS} at once, on every AI tool and on top of Consistency and Fine detail.</small>
      : <small>Put FLUX.2 Klein LoRAs (.safetensors) in the LoRA folder, then press Refresh.</small>)}
    {library?.loras.map(file => {
      const choice = chosen.find(lora => lora.name === file.name);
      const note = loraNote(file, chosen, modelName, locked);
      return <div key={file.name} className="lora-item">
        <label className="quality-switch"><span>{file.name}<small className={file.problem ? "lora-problem" : undefined}>{note ?? megabytes(file.size_bytes)}</small></span>
          <input type="checkbox" role="switch" aria-label={file.name} checked={!!choice} disabled={note !== null}
            onChange={event => choose(file.name, file.models, event.target.checked)} /></label>
        {choice && note === null && <label className="lora-strength">Strength
          <input type="range" min="0.1" max="2" step="0.05" aria-label={`${file.name} strength`} value={choice.scale} onChange={event => scale(file.name, Number(event.target.value))} />
          <output>{choice.scale.toFixed(2)}</output></label>}
      </div>;
    })}
    {library && <div className="lora-folder"><code title={library.folder}>{library.folder}</code>
      <button type="button" onClick={() => void navigator.clipboard.writeText(library.folder)}>Copy path</button></div>}
  </div>;
}
