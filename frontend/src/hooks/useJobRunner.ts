import { MutableRefObject, useRef, useState } from "react";
import { DocumentRecord, JobResponse, JobStatus, cancelJob } from "../api";
import { errorMessage } from "../errors";
import { JobExtras, LayerResult } from "../generation/submission";
import { CancellationRequest, isActiveJob, useJobPolling } from "./useJobPolling";

/** What was submitted, so the finished job knows where its result goes. */
export interface JobContext {
  jobId: string;
  documentId: number | null;
  revisionId: number | null;
  operation: "outpaint" | null;
  layerId: string | null;
  maskLabel: string;
  layerOffset: { x: number; y: number };
  result: LayerResult | null;
}

interface Options {
  documentRef: MutableRefObject<DocumentRecord | null>;
  setError: (message: string | null) => void;
  onDisconnected: () => void;
  /** Called once per completed job; check `isDisposed` after every await. */
  onCompleted: (job: JobStatus, submitted: JobContext | null, isDisposed: () => boolean) => Promise<void>;
}

/** The running engine job: queueing, polling and cancelling. */
export function useJobRunner({ documentRef, setError, onDisconnected, onCompleted }: Options) {
  const [job, setJob] = useState<JobStatus | null>(null);
  const context = useRef<JobContext | null>(null);
  const cancellation = useRef<CancellationRequest | null>(null);
  const active = isActiveJob(job);
  const cancelling = job?.status === "cancelling";

  useJobPolling({
    job, setJob, cancellation, onError: setError, onDisconnected,
    onCompleted: (next, isDisposed) => onCompleted(next, context.current?.jobId === next.id ? context.current : null, isDisposed),
  });

  function queued(created: JobResponse, operation: JobContext["operation"] = null, extras: JobExtras = {}) {
    cancellation.current = null;
    const current = documentRef.current;
    context.current = {
      jobId: created.id, documentId: current?.id ?? null, revisionId: current?.current_revision_id ?? null, operation,
      layerId: extras.layerId ?? null, maskLabel: extras.maskLabel ?? "Cut out object", layerOffset: extras.layerOffset ?? { x: 0, y: 0 }, result: extras.result ?? null,
    };
    setJob({ ...created, progress: 0, error: null, generation_id: null, generation_ids: [] });
  }

  async function cancel() {
    if (!job || !active || cancelling || cancellation.current) return;
    const { id, status } = job;
    cancellation.current = { id, requestedAt: Date.now() };
    setError(null);
    setJob(current => current?.id === id ? { ...current, status: "cancelling" } : current);
    try {
      const next = await cancelJob(id);
      if (cancellation.current?.id === id) setJob(next);
    } catch (err) {
      if (cancellation.current?.id !== id) return;
      cancellation.current = null;
      setJob(current => current?.id === id ? { ...current, status } : current);
      setError("Could not cancel the operation. " + errorMessage(err));
    }
  }

  return { job, setJob, active, cancelling, queued, cancel };
}
