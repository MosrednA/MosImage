import { invoke, isTauri } from "@tauri-apps/api/core";

/**
 * Saves a file where the user picks: a Save As dialog in the desktop app, a download in the browser.
 * Returns where it went, or null when the dialog was cancelled.
 */
export async function saveFile(blob: Blob, name: string): Promise<string | null> {
  if (isTauri()) {
    const bytes = new Uint8Array(await blob.arrayBuffer());
    return invoke<string | null>("save_image", bytes, { headers: { "x-file-name": encodeURIComponent(name) } });
  }
  const href = URL.createObjectURL(blob);
  const anchor = window.document.createElement("a");
  anchor.href = href; anchor.download = name; anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(href), 1000);
  return "your downloads";
}
