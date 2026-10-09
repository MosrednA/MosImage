import { ImageBox } from "../canvas/context";
import { MaskField, cropSelection, distanceField, fieldMargin } from "./maskRefine";
import type { FieldRequest, FieldResponse } from "./maskWorker";

interface Pending { box: ImageBox; selected: Uint8Array; resolve: (field: MaskField) => void }

let worker: Worker | null | undefined;
let nextId = 0;
const pending = new Map<number, Pending>();

/** The shared worker, or null where workers are unavailable (tests) or failed to start. */
function fieldWorker(): Worker | null {
  if (worker !== undefined) return worker;
  try {
    worker = typeof Worker === "undefined" ? null : new Worker(new URL("./maskWorker.ts", import.meta.url), { type: "module" });
  } catch { worker = null; }
  worker?.addEventListener("message", (event: MessageEvent<FieldResponse>) => {
    const request = pending.get(event.data.id);
    if (!request) return;
    pending.delete(event.data.id);
    request.resolve({ box: request.box, signed: event.data.signed });
  });
  // A worker that cannot run hands its requests back to the main thread.
  worker?.addEventListener("error", () => {
    worker?.terminate(); worker = null;
    for (const request of pending.values()) request.resolve(distanceField(request.box, request.selected));
    pending.clear();
  });
  return worker;
}

/** `signedDistanceField`, with the expensive distance passes in a worker. */
export function signedDistanceFieldAsync(rgba: Uint8ClampedArray, imageWidth: number, imageHeight: number, bounds: ImageBox, margin = fieldMargin()): Promise<MaskField> {
  const { box, selected } = cropSelection(rgba, imageWidth, imageHeight, bounds, margin);
  const target = fieldWorker();
  if (!target) return Promise.resolve(distanceField(box, selected));
  return new Promise(resolve => {
    const id = ++nextId;
    // The crop is copied, not transferred, so the main thread can still finish it if the worker fails.
    pending.set(id, { box, selected, resolve });
    target.postMessage({ id, box, selected } satisfies FieldRequest);
  });
}
