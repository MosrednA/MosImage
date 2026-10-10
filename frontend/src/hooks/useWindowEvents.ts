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

/** Keys a focused slider uses itself; every other key stays a shortcut, as in Photoshop. */
const SLIDER_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"]);

/** Typing in a field keeps its keys to itself. */
export function isTyping(event: KeyboardEvent): boolean {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return false;
  if (target.closest("dialog")) return true;
  if (target instanceof HTMLInputElement && target.type === "range") return SLIDER_KEYS.has(event.key);
  return !!target.closest("input, textarea, select, [contenteditable]");
}
