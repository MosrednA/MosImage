import { useEffect, useState } from "react";
import { DocumentRecord, GenerationRecord, ModelInfo, SystemInfo, getSystemInfo, listDocuments, listHistory, listModels } from "../api";
import { useEngineWarmUp } from "./useEngineWarmUp";

/** Python and model imports take a while on a cold start. */
const ENGINE_START_PATIENCE = 90_000;
const ENGINE_RETRY_INTERVAL = 1000;

/** The connection to the local engine, its models and the library lists. */
export function useEngine(setError: (message: string | null) => void) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [online, setOnline] = useState(false);
  const [connecting, setConnecting] = useState(true);
  const [history, setHistory] = useState<GenerationRecord[]>([]);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);

  async function refresh() {
    const [nextHistory, nextDocuments] = await Promise.all([listHistory(), listDocuments()]);
    setHistory(nextHistory); setDocuments(nextDocuments);
  }
  /** Connects to the engine, retrying for `patience` ms while it starts up. */
  async function connect(patience = 0) {
    const deadline = Date.now() + patience;
    setConnecting(true);
    for (;;) {
      try {
        const [nextModels, nextSystem] = await Promise.all([listModels(), getSystemInfo(), refresh()]);
        setModels(nextModels); setSystem(nextSystem); setOnline(true); setError(null);
        break;
      } catch {
        if (Date.now() < deadline) { await new Promise(resolve => window.setTimeout(resolve, ENGINE_RETRY_INTERVAL)); continue; }
        setOnline(false); setError("The local engine is unavailable. Reopen the MosImage app or run start.cmd, then reconnect.");
        break;
      }
    }
    setConnecting(false);
  }
  // The desktop app starts the engine alongside the window, so give it time to come up.
  useEffect(() => { void connect(ENGINE_START_PATIENCE); }, []);
  const warmUp = useEngineWarmUp(online, setModels);
  const cudaBlocked = !!system && (!system.torch_available || (system.preferred_device === "cuda" && !system.cuda_available && !system.allow_cpu_generation));

  return { models, setModels, system, online, setOnline, connecting, connect, warmUp, cudaBlocked, history, documents, setDocuments, refresh };
}
