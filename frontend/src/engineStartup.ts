import { EngineUnreachableError } from "./api";
import { ENGINE_LOG, START_SCRIPT } from "./platform";

/** Python, PyTorch and the model code take a while to import on a cold start. */
export const ENGINE_START_PATIENCE = 90_000;
const FIRST_RETRY = 200;
const MAX_RETRY = 1000;

/** The pause before retry `attempt` (0-based): short at first, so a quick start shows no delay. */
export function retryDelay(attempt: number): number {
  return Math.min(FIRST_RETRY * 2 ** attempt, MAX_RETRY);
}

export interface RetryOptions {
  /** How long to keep trying, in ms; 0 tries once. */
  patience: number;
  /** Stops retrying, for example when a newer attempt to connect began. */
  cancelled?: () => boolean;
  now?: () => number;
  sleep?: (ms: number) => Promise<void>;
}

/**
 * Runs `attempt` until the engine answers, retrying with a short backoff while nothing
 * listens yet. An error the engine itself reports is thrown at once; after `patience`
 * the last connection error is.
 */
export async function untilEngineAnswers<T>(attempt: () => Promise<T>, { patience, cancelled = () => false, now = Date.now, sleep = wait }: RetryOptions): Promise<T> {
  const deadline = now() + patience;
  for (let tries = 0; ; tries++) {
    try { return await attempt(); }
    catch (error) {
      const remaining = deadline - now();
      if (!(error instanceof EngineUnreachableError) || remaining <= 0 || cancelled()) throw error;
      await sleep(Math.min(retryDelay(tries), remaining));
      if (cancelled()) throw error;
    }
  }
}

export interface StartFailure { title: string; detail: string; hint: string }

/** What the start screen says when the engine did not come up; `inApp` is the desktop app, which starts it. */
export function describeStartFailure(error: unknown, inApp: boolean): StartFailure {
  const seconds = Math.round(ENGINE_START_PATIENCE / 1000);
  return {
    title: "MosImage couldn't start its engine",
    detail: error instanceof Error && !(error instanceof EngineUnreachableError)
      ? `The local engine started, but reported a problem: ${error.message}`
      : `The local engine didn't answer within ${seconds} seconds.`,
    hint: inApp
      ? `Close MosImage and open it again. If it still won't start, ${ENGINE_LOG} in the MosImage folder says why.`
      : `Start it with ${START_SCRIPT} in the MosImage folder, then try again.`,
  };
}

function wait(ms: number): Promise<void> {
  return new Promise(resolve => window.setTimeout(resolve, ms));
}
