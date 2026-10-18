import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, EngineUnreachableError, checkHealth, listModels } from "./api";
import { ENGINE_UNREACHABLE } from "./errors";

afterEach(() => { vi.unstubAllGlobals(); });

function fetchFails(error: unknown) {
  vi.stubGlobal("fetch", vi.fn(async () => { throw error; }));
}

describe("requests to the engine", () => {
  it("turns no answer into a clear error instead of \"Failed to fetch\"", async () => {
    fetchFails(new TypeError("Failed to fetch"));
    const failure = await listModels().catch(error => error);
    expect(failure).toBeInstanceOf(EngineUnreachableError);
    expect(failure.message).toBe(ENGINE_UNREACHABLE);
  });

  it("counts a health check that times out as no answer", async () => {
    fetchFails(new DOMException("The operation timed out.", "TimeoutError"));
    await expect(checkHealth()).rejects.toBeInstanceOf(EngineUnreachableError);
  });

  it("leaves a deliberate abort alone", async () => {
    fetchFails(new DOMException("The operation was aborted.", "AbortError"));
    await expect(listModels()).rejects.toMatchObject({ name: "AbortError" });
  });

  it("keeps the engine's own error and status", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "Database is locked" }), { status: 500 })));
    const failure = await listModels().catch(error => error);
    expect(failure).toBeInstanceOf(ApiError);
    expect(failure).toMatchObject({ message: "Database is locked", status: 500 });
  });

  it("asks the health endpoint whether the engine is up", async () => {
    const fetcher = vi.fn(async (_url: string, _init?: RequestInit) => new Response(JSON.stringify({ status: "ok" })));
    vi.stubGlobal("fetch", fetcher);
    await checkHealth();
    expect(fetcher.mock.calls[0][0]).toMatch(/\/api\/health$/);
    expect(fetcher.mock.calls[0][1]?.signal).toBeInstanceOf(AbortSignal);
  });
});
