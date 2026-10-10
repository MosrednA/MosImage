/** The UI runs in WebView2 on Windows and in WebKit on macOS, in the app and in a browser. */
export const IS_MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);

/** The shortcut modifier as the keyboard labels it; the handlers accept Ctrl and ⌘ on both. */
export const MOD = IS_MAC ? "⌘" : "Ctrl";

/** How to start the engine by hand when the desktop app is not running. */
export const START_SCRIPT = IS_MAC ? "./start.sh" : "start.cmd";
