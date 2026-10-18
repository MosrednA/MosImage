import { useEffect, useRef, useState } from "react";
import { DocumentRecord, GenerationRecord, ModelInfo, SystemInfo, checkHealth, getSystemInfo, listDocuments, listHistory, listModels } from "../api";
import { ENGINE_START_PATIENCE, StartFailure, describeStartFailure, untilEngineAnswers } from "../engineStartup";
import { applyPresetModels } from "../generation/presets";
import { IN_APP, START_SCRIPT } from "../platform";
import { useEngineWarmUp } from "./useEngineWarmUp";

/** The connection to the local engine, its models and the library lists. */
export function useEngine(setError: (message: string | null) => void) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [online, setOnline] = useState(false);
  const [connecting, setConnecting] = useState(true);
  const [history, setHistory] = useState<GenerationRecord[]>([]);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  // The start screen shows until the engine first answers; losing it later shows an error instead.
  const [started, setStarted] = useState(false);
  const [startFailure, setStartFailure] = useState<StartFailure | null>(null);
  const startedRef = useRef(false);
  // A newer connect, or closing the window, stops an older one that is still retrying.
  const attempt = useRef(0);

  async function refresh() {
    const [nextHistory, nextDocuments] = await Promise.all([listHistory(), listDocuments()]);
    setHistory(nextHistory); setDocuments(nextDocuments);
  }
  /** Connects to the engine, retrying for `patience` ms while it starts up. */
  async function connect(patience = 0) {
    const current = ++attempt.current;
    const cancelled = () => attempt.current !== current;
    setConnecting(true); setStartFailure(null);
    try {
      const [nextModels, nextSystem] = await untilEngineAnswers(async () => {
        await checkHealth();
        return Promise.all([listModels(), getSystemInfo(), refresh()]);
      }, { patience, cancelled });
      if (cancelled()) return;
      applyPresetModels(nextModels);
      setModels(nextModels); setSystem(nextSystem); setOnline(true); setError(null);
      startedRef.current = true; setStarted(true);
    } catch (err) {
      if (cancelled()) return;
      setOnline(false);
      if (startedRef.current) setError(`The local engine is unavailable. Reopen the MosImage app or run ${START_SCRIPT}, then reconnect.`);
      else setStartFailure(describeStartFailure(err, IN_APP));
    }
    setConnecting(false);
  }
  /** Waits for the engine again from the start screen, as on opening the app. */
  function retryStart() { void connect(ENGINE_START_PATIENCE); }
  // The desktop app starts the engine alongside the window, so give it time to come up.
  useEffect(() => {
    void connect(ENGINE_START_PATIENCE);
    return () => { attempt.current++; };
  }, []);
  const warmUp = useEngineWarmUp(online, setModels);
  // The engine runs, but PyTorch cannot reach the GPU it is set up for: CUDA, or Metal on a Mac.
  const deviceBlocked = !!system && (!system.torch_available
    || (system.preferred_device !== "cpu" && system.selected_device === "cpu" && !system.allow_cpu_generation));

  return { models, setModels, system, online, setOnline, connecting, connect, started, startFailure, retryStart, warmUp, deviceBlocked, history, documents, setDocuments, refresh };
}
