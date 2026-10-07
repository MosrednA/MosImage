import { CSSProperties } from "react";
import { JobStatus } from "../api";
import { describeJob } from "../generation/progress";

export function JobOverlay({ job }: { job: JobStatus | null }) {
  const text = job ? describeJob(job) : { title: "Opening image…", detail: "", percent: null };
  const items = job?.items ?? 1;
  return <div className="job-overlay" role="status">
    <span className="spinner" />
    <div><strong>{text.title}</strong>{text.detail && <small>{text.detail}</small>}</div>
    {text.percent !== null && <output>{text.percent}%</output>}
    {text.percent !== null && <span className="job-progress" aria-hidden="true">
      {/* One segment per proposal, so two proposals read as two halves. */}
      {Array.from({ length: items }, (_, index) => <i key={index} style={{ "--fill": `${segmentFill(job!.progress, index, items) * 100}%` } as CSSProperties} />)}
    </span>}
  </div>;
}

function segmentFill(progress: number, index: number, count: number): number {
  return Math.min(Math.max(progress * count - index, 0), 1);
}
