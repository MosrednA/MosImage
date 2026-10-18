import { describe, expect, it } from "vitest";
import { ApiError, EngineUnreachableError } from "./api";
import { ENGINE_START_PATIENCE, describeStartFailure, retryDelay, untilEngineAnswers } from "./engineStartup";
import { START_SCRIPT } from "./platform";

/** A fake clock that sleeping moves on, and the pauses taken. */
function clock() {
  let time = 0;
  const sleeps: number[] = [];
  return { sleeps, now: () => time, sleep: async (ms: number) => { sleeps.push(ms); time += ms; } };
}

/** Fails with `errors` in turn, then answers "ready". */
function engine(...errors: Error[]) {
  let calls = 0;
  const attempt = async () => {
    calls++;
    const error = errors.shift();
    if (error) throw error;
    return "ready";
  };
  return { attempt, calls: () => calls };
}

const unreachable = (count: number) => Array.from({ length: count }, () => new EngineUnreachableError());

describe("engine start-up retries", () => {
  it("backs off from a short first pause to at most a second", () => {
    expect([0, 1, 2, 3, 4, 10].map(retryDelay)).toEqual([200, 400, 800, 1000, 1000, 1000]);
  });

  it("answers at once when the engine is already up", async () => {
    const time = clock();
    const started = engine();
    await expect(untilEngineAnswers(started.attempt, { patience: ENGINE_START_PATIENCE, ...time })).resolves.toBe("ready");
    expect(started.calls()).toBe(1);
    expect(time.sleeps).toEqual([]);
  });

  it("retries quietly while nothing listens yet", async () => {
    const time = clock();
    const started = engine(...unreachable(4));
    await expect(untilEngineAnswers(started.attempt, { patience: ENGINE_START_PATIENCE, ...time })).resolves.toBe("ready");
    expect(started.calls()).toBe(5);
    expect(time.sleeps).toEqual([200, 400, 800, 1000]);
  });

  it("reports an error from the engine itself without waiting", async () => {
    const time = clock();
    const started = engine(new ApiError("Database is locked", 500));
    await expect(untilEngineAnswers(started.attempt, { patience: ENGINE_START_PATIENCE, ...time })).rejects.toThrow("Database is locked");
    expect(started.calls()).toBe(1);
    expect(time.sleeps).toEqual([]);
  });

  it("gives up once its patience runs out, with one last try at the deadline", async () => {
    const time = clock();
    const started = engine(...unreachable(10));
    await expect(untilEngineAnswers(started.attempt, { patience: 1500, ...time })).rejects.toBeInstanceOf(EngineUnreachableError);
    expect(time.sleeps).toEqual([200, 400, 800, 100]);
    expect(time.now()).toBe(1500);
    expect(started.calls()).toBe(5);
  });

  it("tries once without patience, as Reconnect does", async () => {
    const time = clock();
    const started = engine(...unreachable(1));
    await expect(untilEngineAnswers(started.attempt, { patience: 0, ...time })).rejects.toBeInstanceOf(EngineUnreachableError);
    expect(started.calls()).toBe(1);
  });

  it("stops when a newer attempt takes over", async () => {
    const time = clock();
    const started = engine(...unreachable(10));
    const cancelled = () => started.calls() >= 2;
    await expect(untilEngineAnswers(started.attempt, { patience: ENGINE_START_PATIENCE, cancelled, ...time })).rejects.toBeInstanceOf(EngineUnreachableError);
    expect(started.calls()).toBe(2);
  });
});

describe("start-up failure wording", () => {
  it("says how long it waited when nothing answered", () => {
    const failure = describeStartFailure(new EngineUnreachableError(), true);
    expect(failure.detail).toBe("The local engine didn't answer within 90 seconds.");
    expect(failure.hint).toMatch(/engine\.log in the MosImage folder/);
    expect(JSON.stringify(failure)).not.toMatch(/fetch/i);
  });

  it("passes on what the engine reported", () => {
    expect(describeStartFailure(new ApiError("Database is locked", 500), true).detail).toBe("The local engine started, but reported a problem: Database is locked");
  });

  it("points a browser at the start script", () => {
    expect(describeStartFailure(new EngineUnreachableError(), false).hint).toBe(`Start it with ${START_SCRIPT} in the MosImage folder, then try again.`);
  });
});
