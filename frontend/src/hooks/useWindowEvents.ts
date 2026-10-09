import { useEffect, useRef } from "react";

interface Handlers {
  onKey: (event: KeyboardEvent) => void;
  onPaste: (event: ClipboardEvent) => void;
}

/** Window-wide shortcuts and paste; the latest handlers always run, so they never see stale state. */
export function useWindowEvents(handlers: Handlers) {
  const latest = useRef(handlers);
  latest.current = handlers;
  useEffect(() => {
    const keys = (event: KeyboardEvent) => latest.current.onKey(event);
    const paste = (event: ClipboardEvent) => latest.current.onPaste(event);
    window.addEventListener("keydown", keys); window.addEventListener("paste", paste);
    return () => { window.removeEventListener("keydown", keys); window.removeEventListener("paste", paste); };
  }, []);
}

/** Typing in a field keeps its keys to itself. */
export function isTyping(event: KeyboardEvent): boolean {
  return event.target instanceof HTMLElement && !!event.target.closest("input, textarea, select, [contenteditable], dialog");
}
