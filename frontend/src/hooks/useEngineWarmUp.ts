import { useEffect, useRef, useState } from "react";
import { JobStatus, ModelInfo, getEngine } from "../api";
import { isActiveJob } from "./useJobPolling";

const POLL_INTERVAL = 1000;

/** Follows the start-up warm-up until the Quality model and SAM are loaded. */
export function useEngineWarmUp(online: boolean, onModels: (models: ModelInfo[]) => void): JobStatus | null {
  const [warmUp, setWarmUp] = useState<JobStatus | null>(null);
  const handler = useRef(onModels);
  handler.current = onModels;

  useEffect(() => {
    if (!online) return;
    let disposed = false, timer = 0;
    async function poll() {
      try {
        const engine = await getEngine();
        if (disposed) return;
        setWarmUp(engine.warm_up);
        if (isActiveJob(engine.warm_up)) { timer = window.setTimeout(poll, POLL_INTERVAL); return; }
        handler.current(engine.models);
      } catch {
        // Older engines lack /api/engine; models then simply load on first use.
        if (!disposed) setWarmUp(null);
      }
    }
    void poll();
    return () => { disposed = true; window.clearTimeout(timer); };
  }, [online]);

  return warmUp;
}
