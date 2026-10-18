import { isTauri } from "@tauri-apps/api/core";

/** The UI runs in WebView2 on Windows and in WebKit on macOS, in the app and in a browser. */
export const IS_MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);

/** The desktop app, which starts the engine itself; in a browser, the start script does. */
export const IN_APP = isTauri();

/** The Windows app has no native title bar (tauri.windows.conf.json), so the top bar is one; the Mac keeps its own. */
export const OWN_TITLE_BAR = IN_APP && !IS_MAC;

/** The shortcut modifier as the keyboard labels it; the handlers accept Ctrl and ⌘ on both. */
export const MOD = IS_MAC ? "⌘" : "Ctrl";

/** How to start the engine by hand when the desktop app is not running. */
export const START_SCRIPT = IS_MAC ? "./start.sh" : "start.cmd";

/** Where the desktop app writes the engine's output, in the MosImage folder. */
export const ENGINE_LOG = IS_MAC ? "data/logs/engine.log" : "data\\logs\\engine.log";

/** The Alt key as the keyboard labels it: Option (⌥) on a Mac. */
export const ALT = IS_MAC ? "⌥" : "Alt";
