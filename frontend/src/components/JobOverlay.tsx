import { JobStatus } from "../api";

export function JobOverlay({ job }: { job: JobStatus | null }) {
  const cancelling = job?.status === "cancelling";
  const detail = cancelling ? "Stopping the operation and releasing memory…"
    : job?.kind === "select" ? "The selection model downloads on first use"
      : (job?.progress ?? 0) <= .05 ? "The first run may take longer while the model loads"
        : "Your preview will appear as soon as it is ready";
  return <div className="job-overlay" role="status">
    <span className="spinner" />
    <div><strong>{job ? jobLabel(job) : "Opening image…"}</strong><small>{detail}</small></div>
    {job && !cancelling && <output>{Math.round(job.progress * 100)}%</output>}
  </div>;
}

function jobLabel(job: JobStatus): string {
  if (job.status === "cancelling") return "Cancelling…";
  if (job.status === "queued") return "Waiting for the local engine…";
  if (job.kind === "select") return "Selecting object…";
  return job.progress <= .05 ? "Loading model…" : job.kind === "outpaint" ? "Expanding your image…" : "Creating your proposals…";
}
