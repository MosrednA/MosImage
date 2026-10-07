import { Dispatch, MutableRefObject, SetStateAction, useEffect, useRef } from "react";
import { ApiError, JobStatus, getJob } from "../api";

export interface CancellationRequest { id: string; requestedAt: number }

interface Options {
  job: JobStatus | null;
  setJob: Dispatch<SetStateAction<JobStatus | null>>;
  cancellation: MutableRefObject<CancellationRequest | null>;
  /** Called once per completed job; check `isDisposed` after every await. */
  onCompleted: (job: JobStatus, isDisposed: () => boolean) => Promise<void>;
  onError: (message: string) => void;
  onDisconnected: () => void;
}

const POLL_INTERVAL = 800;
const CANCEL_RECONNECT_WINDOW = 30_000;

export function isActiveJob(job: JobStatus | null): boolean {
  return job?.status === "queued" || job?.status === "running" || job?.status === "cancelling";
}

export function useJobPolling({ job, setJob, cancellation, onCompleted, onError, onDisconnected }: Options) {
  const handlers = useRef({ onCompleted, onError, onDisconnected });
  handlers.current = { onCompleted, onError, onDisconnected };

  useEffect(() => {
    if (!job) return;
    const id = job.id;
    let disposed = false, timer = 0;
    const isDisposed = () => disposed;

    async function poll() {
      try {
        const next = await getJob(id);
        if (disposed) return;
        // A poll already in flight can return an older running state after Cancel.
        if (cancellation.current?.id === id && (next.status === "queued" || next.status === "running")) next.status = "cancelling";
        if (next.status === "completed") await handlers.current.onCompleted(next, isDisposed);
        if (disposed) return;
        setJob(next);
        if (next.status === "cancelled" || next.status === "completed" || next.status === "failed") cancellation.current = null;
        if (next.status === "failed") handlers.current.onError(next.error ?? "The operation failed.");
        if (isActiveJob(next)) timer = window.setTimeout(poll, POLL_INTERVAL);
      } catch (err) {
        if (disposed) return;
        if (cancellation.current?.id === id) {
          // A supervised hard cancel restarts the worker and removes its in-memory jobs.
          if (err instanceof ApiError && err.status === 404) {
            cancellation.current = null;
            setJob(current => current?.id === id ? { ...current, status: "cancelled" } : current);
            return;
          }
          if (Date.now() - cancellation.current.requestedAt < CANCEL_RECONNECT_WINDOW) {
            timer = window.setTimeout(poll, POLL_INTERVAL);
            return;
          }
          handlers.current.onDisconnected();
          handlers.current.onError("Cancellation was requested, but the local engine has not reconnected. Reconnect to continue.");
        } else handlers.current.onError(err instanceof Error ? err.message : "The operation failed.");
        cancellation.current = null;
        setJob(current => current?.id === id ? { ...current, status: "failed" } : current);
      }
    }

    timer = window.setTimeout(poll, 400);
    return () => { disposed = true; window.clearTimeout(timer); };
  }, [job?.id]);
}
