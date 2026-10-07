import { describe, expect, it } from "vitest";
import { JobStatus } from "../api";
import { describeJob, duration } from "./progress";

function job(changes: Partial<JobStatus>): JobStatus {
  return { id: "1", kind: "inpaint", status: "running", progress: .5, error: null, generation_id: null, generation_ids: [], ...changes };
}

describe("job progress wording", () => {
  it("names the model and elapsed time while loading", () => {
    const text = describeJob(job({ phase: "loading", detail: "FLUX.2 Klein 4B NVFP4", phase_elapsed_seconds: 6.4 }));
    expect(text.title).toBe("Loading FLUX.2 Klein 4B NVFP4…");
    expect(text.detail).toMatch(/^6 s · usually/);
  });

  it("shows the proposal, step and time left while sampling", () => {
    const text = describeJob(job({ phase: "generating", item: 2, items: 2, step: 3, steps: 4, eta_seconds: 7.6, progress: .72 }));
    expect(text).toEqual({ title: "Creating proposal 2 of 2…", detail: "Step 3 of 4 · about 8 s left", percent: 72 });
  });

  it("says the prompt is being read before the first step", () => {
    expect(describeJob(job({ phase: "generating", item: 1, items: 1 })).detail).toBe("Preparing your image");
    expect(describeJob(job({ phase: "preparing", item: 1, items: 1 })).detail).toBe("Reading your description");
  });

  it("explains a wait behind the start-up warm-up", () => {
    const text = describeJob(job({ status: "queued", waiting_for: "load-model" }));
    expect(text.title).toBe("Waiting for the model to warm up…");
    expect(text.percent).toBeNull();
  });

  it("formats long estimates in minutes", () => {
    expect(duration(83)).toBe("1 min 23 s");
    expect(duration(120)).toBe("2 min");
    expect(duration(.3)).toBe("1 s");
  });
});
