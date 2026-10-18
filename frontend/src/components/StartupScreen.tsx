import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { StartFailure } from "../engineStartup";
import { handOffSplash } from "../splash";
import { Icon } from "./Icon";

/** After this the start screen explains that a cold start takes longer. */
const SLOW_START = 15_000;

/** Shown below the top bar until the local engine first answers, while the desktop app starts it. */
export function StartupScreen({ failure, onRetry }: { failure: StartFailure | null; onRetry: () => void }) {
  const [slow, setSlow] = useState(false);
  const markRef = useRef<HTMLDivElement>(null);
  // Before the first paint, so the loading logo and this one are never both on screen.
  useLayoutEffect(() => handOffSplash(markRef.current), []);
  useEffect(() => {
    setSlow(false);
    if (failure) return;
    const timer = window.setTimeout(() => setSlow(true), SLOW_START);
    return () => window.clearTimeout(timer);
  }, [failure]);

  const mark = <div ref={markRef} className="startup-mark" aria-hidden="true">m<span>•</span></div>;
  if (failure) return <section className="startup-screen failed" role="alert">
    {mark}
    <h1>{failure.title}</h1>
    <p>{failure.detail}</p>
    <p className="startup-hint">{failure.hint}</p>
    <button className="primary" autoFocus onClick={onRetry}><Icon name="reset" />Try again</button>
  </section>;
  return <section className="startup-screen" role="status">
    {mark}
    <h1>Starting MosImage…</h1>
    <p>{slow ? "Still starting. After a restart or an update, this can take up to a minute." : "Getting the local engine ready. This takes a few seconds."}</p>
    <div className="startup-progress" aria-hidden="true"><i /></div>
  </section>;
}
