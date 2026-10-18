/** The logo index.html shows centred while the app loads (#splash there), handed to the start screen. */

/** How long the logo glides into place. */
export const GLIDE_MS = 450;
/** The least time the logo sits centred first, so a fast load still shows where it came from. */
export const HOLD_MS = 300;
/** How fast the logo fades when there is no start screen to glide to. */
const FADE_MS = 160;
/** After this past an animation's end its cleanup runs anyway: animations wait for frames, which a window hidden at start does not draw. */
const SPARE_MS = 500;

type Box = Pick<DOMRect, "left" | "top" | "width" | "height">;

/** The move that puts the centre of `from` on the centre of `to`. */
export function glide(from: Box, to: Box): { x: number; y: number } {
  return { x: to.left + to.width / 2 - (from.left + from.width / 2), y: to.top + to.height / 2 - (from.top + from.height / 2) };
}

/** How much longer the logo sits still, `elapsed` ms after the page started loading. */
export function holdLeft(elapsed: number): number {
  return Math.max(0, Math.round(HOLD_MS - elapsed));
}

/**
 * Glide and shrink the loading logo onto `target`, the start screen's own logo, which stays hidden
 * until it lands, while the background fades to show the screen around it. Returns a cleanup for the start
 * screen's effect: if the screen goes before the logo lands (the engine was already running), the
 * logo fades out instead. Under StrictMode's trial remount the target stays connected, so nothing happens.
 */
export function handOffSplash(target: HTMLElement | null): () => void {
  const splash = document.getElementById("splash");
  if (splash && !splash.dataset.leaving) {
    splash.dataset.leaving = "true";
    const logo = splash.firstElementChild;
    const still = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (!target || !(logo instanceof HTMLElement) || still || typeof logo.animate !== "function") fadeOut(splash);
    else {
      const { x, y } = glide(logo.getBoundingClientRect(), target.getBoundingClientRect());
      const hold = holdLeft(performance.now());
      const timing: KeyframeAnimationOptions = { duration: GLIDE_MS, delay: hold, easing: "cubic-bezier(.2, .8, .2, 1)", fill: "forwards" };
      target.style.visibility = "hidden";
      // From index.html's larger centred logo to the start screen's, which has its own size.
      logo.animate([{ transform: getComputedStyle(logo).transform }, { transform: `translate(${x}px, ${y}px)` }], timing);
      const landed = () => { target.style.visibility = ""; splash.remove(); };
      splash.animate([{ backgroundColor: "#1c2027" }, { backgroundColor: "#1c202700" }], timing).finished.then(landed, landed);
      window.setTimeout(landed, hold + GLIDE_MS + SPARE_MS);
    }
  }
  return () => window.setTimeout(() => {
    const left = document.getElementById("splash");
    if (left && target && !target.isConnected) fadeOut(left);
  });
}

function fadeOut(splash: HTMLElement): void {
  if (splash.dataset.fading) return;
  splash.dataset.fading = "true";
  if (typeof splash.animate !== "function") { splash.remove(); return; }
  const gone = () => splash.remove();
  splash.animate([{ opacity: 1 }, { opacity: 0 }], { duration: FADE_MS, fill: "forwards" }).finished.then(gone, gone);
  window.setTimeout(gone, FADE_MS + SPARE_MS);
}
