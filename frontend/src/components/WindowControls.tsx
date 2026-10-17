import { useEffect, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";

/** Minimize, maximize and close, drawn like Windows 11 caption buttons for the frameless Windows app. */
export function WindowControls() {
  const [maximized, setMaximized] = useState(false);
  useEffect(() => {
    const appWindow = getCurrentWindow();
    let live = true, stop: (() => void) | null = null;
    const sync = () => void appWindow.isMaximized().then(value => { if (live) setMaximized(value); });
    sync();
    void appWindow.onResized(sync).then(unlisten => { if (live) stop = unlisten; else unlisten(); });
    return () => { live = false; stop?.(); };
  }, []);
  const appWindow = getCurrentWindow();
  return <div className="window-controls">
    <button aria-label="Minimize" title="Minimize" tabIndex={-1} onClick={() => void appWindow.minimize()}>
      <svg viewBox="0 0 10 10" aria-hidden="true"><path d="M0 5.5h10" /></svg>
    </button>
    <button aria-label={maximized ? "Restore Down" : "Maximize"} title={maximized ? "Restore Down" : "Maximize"} tabIndex={-1} onClick={() => void appWindow.toggleMaximize()}>
      {maximized
        ? <svg viewBox="0 0 10 10" aria-hidden="true"><path d="M2.5 2.5V.5h7v7h-2M.5 2.5h7v7h-7z" /></svg>
        : <svg viewBox="0 0 10 10" aria-hidden="true"><path d="M.5 .5h9v9h-9z" /></svg>}
    </button>
    <button className="close" aria-label="Close" title="Close" tabIndex={-1} onClick={() => void appWindow.close()}>
      <svg viewBox="0 0 10 10" aria-hidden="true"><path d="M0 0l10 10M10 0L0 10" /></svg>
    </button>
  </div>;
}
