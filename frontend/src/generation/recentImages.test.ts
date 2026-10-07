import { describe, expect, it, vi } from "vitest";

import { GenerationRecord } from "../api";
import { generationAsFile } from "./recentImages";

const record = {
  id: 42,
  task: "inpaint",
  output_image_url: "/api/files/output.png",
} as GenerationRecord;

describe("recent image loading", () => {
  it("turns a generated output into an uploadable source file", async () => {
    const fetcher = vi.fn(async () => new Response(new Blob(["image"], { type: "image/png" })));

    const file = await generationAsFile(record, "http://localhost/output.png", fetcher);

    expect(fetcher).toHaveBeenCalledWith("http://localhost/output.png");
    expect(file.name).toBe("generation-42.png");
    expect(file.type).toBe("image/png");
  });

  it("reports a missing generated file", async () => {
    const fetcher = vi.fn(async () => new Response(null, { status: 404 }));

    await expect(generationAsFile(record, "http://localhost/missing.png", fetcher)).rejects.toThrow(
      "Could not open generation 42: 404",
    );
  });
});
