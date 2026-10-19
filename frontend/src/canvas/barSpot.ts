import { CSSProperties } from "react";

/**
 * Where the user put a floating bar: its share of the free room across and
 * down the canvas, 0–1, so it keeps to its corner when the window resizes.
 */
export interface BarSpot { x: number; y: number }

interface Size { width: number; height: number }

/** Keeps the bar clear of the canvas edges and their 10 px scrollbars. */
export const BAR_MARGIN = 12;

/** The spot for a bar whose top-left corner lands at `left`, `top` in the frame. */
export function spotAt(left: number, top: number, bar: Size, frame: Size): BarSpot {
  return { x: share(left, bar.width, frame.width), y: share(top, bar.height, frame.height) };
}

function share(offset: number, size: number, room: number): number {
  const free = room - size - 2 * BAR_MARGIN;
  return free > 0 ? Math.min(Math.max((offset - BAR_MARGIN) / free, 0), 1) : 0;
}

/** Places the bar at its spot without measuring it: the translate is relative to the bar's own size. */
export function spotStyle({ x, y }: BarSpot): CSSProperties {
  return {
    left: `calc(${BAR_MARGIN}px + (100% - ${2 * BAR_MARGIN}px) * ${x})`,
    top: `calc(${BAR_MARGIN}px + (100% - ${2 * BAR_MARGIN}px) * ${y})`,
    transform: `translate(${-x * 100}%, ${-y * 100}%)`,
  };
}

/** A stored spot, or null when there is none or it is unreadable. */
export function parseSpot(text: string | null): BarSpot | null {
  if (!text) return null;
  try {
    const { x, y } = JSON.parse(text) as Partial<BarSpot>;
    if (typeof x !== "number" || typeof y !== "number" || !Number.isFinite(x) || !Number.isFinite(y)) return null;
    return { x: Math.min(Math.max(x, 0), 1), y: Math.min(Math.max(y, 0), 1) };
  } catch {
    return null;
  }
}
