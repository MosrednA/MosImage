/** What a request says when nothing answered, instead of the browser's "Failed to fetch". */
export const ENGINE_UNREACHABLE = "Could not reach the local engine. Try again in a moment; if it keeps happening, reopen MosImage.";

/** fetch's own words when no answer came: WebView2 and Chrome, WebKit, Firefox. */
const FETCH_FAILURE = /^(Failed to fetch|Load failed|NetworkError when attempting to fetch resource)/;

export function errorMessage(error: unknown): string {
  if (error instanceof TypeError && FETCH_FAILURE.test(error.message)) return ENGINE_UNREACHABLE;
  return error instanceof Error ? error.message : "The operation failed.";
}
