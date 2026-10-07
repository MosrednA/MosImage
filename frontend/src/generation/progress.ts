import { JobStatus } from "../api";

export interface JobDescription {
  title: string;
  detail: string;
  /** 0–100, or null while the job has no meaningful progress yet. */
  percent: number | null;
}

/** Human wording for what a job is doing right now. */
export function describeJob(job: JobStatus): JobDescription {
  const percent = job.status === "running" ? Math.round(job.progress * 100) : null;
  if (job.status === "cancelling") return { title: "Cancelling…", detail: "Stopping the operation and releasing memory", percent: null };
  if (job.status === "queued") {
    return job.waiting_for === "load-model"
      ? { title: "Waiting for the model to warm up…", detail: "This happens once after starting MosImage, then edits start right away", percent: null }
      : { title: job.waiting_for ? "Waiting for the current job…" : "Starting…", detail: "Your request is next in line", percent: null };
  }
  const elapsed = job.phase_elapsed_seconds ?? job.elapsed_seconds ?? 0;
  if (job.kind === "select") {
    return job.phase === "loading"
      ? { title: "Loading the selection model…", detail: `${job.detail ?? "SAM"} · ${seconds(elapsed)} · downloads on first use`, percent }
      : { title: "Selecting object…", detail: "Finding the edges around your points", percent };
  }
  if (job.phase === "loading") {
    return { title: `Loading ${job.detail ?? "the model"}…`, detail: `${seconds(elapsed)} · usually 5–15 s, then it stays loaded for your next edits`, percent };
  }
  if (job.phase === "warming") {
    return { title: `Warming up ${job.detail ?? "the model"}…`, detail: "One-time GPU preparation so your first edit is fast", percent };
  }
  const subject = (job.items ?? 1) > 1 ? `Creating proposal ${job.item} of ${job.items}`
    : job.kind === "outpaint" ? "Expanding your image" : job.kind === "text-to-image" ? "Creating your image"
      : job.kind === "load-model" ? "Preparing the model" : "Creating your edit";
  if (job.phase === "finishing") return { title: subject + "…", detail: "Blending into your image" + remaining(job), percent };
  if (job.phase === "generating") {
    const step = job.step && job.steps ? `Step ${job.step} of ${job.steps}` : "Preparing your image";
    return { title: subject + "…", detail: step + remaining(job), percent };
  }
  if (job.phase === "preparing") return { title: subject + "…", detail: "Reading your description" + remaining(job), percent };
  return { title: subject + "…", detail: "Preparing your image", percent };
}

function remaining(job: JobStatus): string {
  const eta = job.eta_seconds;
  return eta != null && eta >= 1 ? ` · about ${duration(eta)} left` : "";
}

function seconds(value: number): string { return `${Math.round(value)} s`; }

export function duration(value: number): string {
  const total = Math.max(1, Math.round(value));
  if (total < 60) return `${total} s`;
  const minutes = Math.floor(total / 60), rest = total % 60;
  return rest ? `${minutes} min ${rest} s` : `${minutes} min`;
}
