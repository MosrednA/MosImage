export type IconName = "object" | "brush" | "expand" | "move" | "library" | "upload" | "undo" | "redo" | "download" | "spark" | "close" | "settings";

const paths: Record<IconName, string> = {
  object: "M8 3H3v5m13-5h5v5M3 16v5h5m8 0h5v-5M8 9l4-2 4 2v6l-4 2-4-2z",
  brush: "M14 5l5-2 2 2-2 5-9 9-5-5zM5 14c-4 1-2 5-4 7 5 0 7-1 7-4",
  expand: "M8 3H3v5m13-5h5v5M3 16v5h5m8 0h5v-5M9 9h6v6H9z",
  move: "M12 2v20M2 12h20M8 6l4-4 4 4M8 18l4 4 4-4M6 8l-4 4 4 4M18 8l4 4-4 4",
  library: "M3 4h7l2 3h9v13H3zM7 11h10M7 15h6",
  upload: "M12 16V3M7 8l5-5 5 5M4 15v6h16v-6",
  undo: "M8 4L3 9l5 5M3 9h10a7 7 0 010 14",
  redo: "M16 4l5 5-5 5M21 9h-10a7 7 0 000 14",
  download: "M12 3v13M7 11l5 5 5-5M4 17v4h16v-4",
  spark: "M12 3l2.4 6.6L21 12l-6.6 2.4L12 21l-2.4-6.6L3 12l6.6-2.4z",
  close: "M6 6l12 12M18 6L6 18",
  settings: "M4 6h16M4 12h16M4 18h16M8 3v6M16 9v6M10 15v6",
};

export function Icon({ name }: { name: IconName }) {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>;
}
