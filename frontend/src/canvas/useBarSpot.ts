import { PointerEvent as ReactPointerEvent, RefObject, useEffect, useRef, useState } from "react";
import { BarSpot, parseSpot, spotAt } from "./barSpot";

const SPOT_KEY = "mosimage.selectionActionsSpot";
/** A grab that moves less than this is a click, which leaves the bar where it is. */
const DRAG_THRESHOLD = 3;

export interface BarGrip {
  /** The bar sits where the user put it instead of beside the selection. */
  moved: boolean;
  dragging: boolean;
  onPointerDown: (event: ReactPointerEvent<HTMLElement>) => void;
  /** Puts the bar back beside the selection. */
  onReset: () => void;
}

/**
 * Lets the user drag a floating bar by its grip. Once moved, the bar stays
 * at that spot in the canvas frame, through pans, zooms, new selections and
 * restarts, until a double-click on the grip puts it back.
 */
export function useBarSpot(frameRef: RefObject<HTMLElement | null>) {
  const [spot, setSpot] = useState<BarSpot | null>(() => { try { return parseSpot(localStorage.getItem(SPOT_KEY)); } catch { return null; } });
  const [dragging, setDragging] = useState(false);
  const stopRef = useRef<(() => void) | null>(null);
  useEffect(() => () => stopRef.current?.(), []);

  function save(next: BarSpot | null) {
    try {
      if (next) localStorage.setItem(SPOT_KEY, JSON.stringify(next));
      else localStorage.removeItem(SPOT_KEY);
    } catch { /* Preference only. */ }
  }

  function onPointerDown(event: ReactPointerEvent<HTMLElement>) {
    const frame = frameRef.current, bar = event.currentTarget.closest<HTMLElement>("[role=toolbar]");
    if (event.button !== 0 || !frame || !bar) return;
    // Keeps the focus where it was, so a prompt being typed stays active.
    event.preventDefault();
    stopRef.current?.();
    const startX = event.clientX, startY = event.clientY;
    const from = bar.getBoundingClientRect(), box = frame.getBoundingClientRect();
    const barSize = { width: from.width, height: from.height }, frameSize = { width: frame.clientWidth, height: frame.clientHeight };
    let moved: BarSpot | null = null;
    // The bar leaves the image for the frame on the first move, which unmounts
    // the grip, so the gesture is followed on the window instead of captured.
    const move = (pointer: PointerEvent) => {
      const dx = pointer.clientX - startX, dy = pointer.clientY - startY;
      if (!moved && Math.hypot(dx, dy) < DRAG_THRESHOLD) return;
      if (!moved) setDragging(true);
      moved = spotAt(from.left - box.left + dx, from.top - box.top + dy, barSize, frameSize);
      setSpot(moved);
    };
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
      stopRef.current = null;
      setDragging(false);
      if (moved) save(moved);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
    window.addEventListener("pointercancel", stop);
    stopRef.current = stop;
  }

  function onReset() {
    setSpot(null);
    save(null);
  }

  const grip: BarGrip = { moved: spot !== null, dragging, onPointerDown, onReset };
  return { spot, grip };
}
