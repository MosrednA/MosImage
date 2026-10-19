/**
 * Height of the prompt box. At rest it keeps its one-line height; while the
 * user types in it, it grows to fit the text (CSS caps it and scrolls beyond).
 */
export function promptHeight(scrollHeight: number, border: number, rest: number, expanded: boolean): number {
  return expanded ? Math.max(rest, Math.ceil(scrollHeight + border)) : rest;
}
