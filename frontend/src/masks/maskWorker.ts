import { ImageBox } from "../canvas/context";
import { distanceField } from "./maskRefine";

export interface FieldRequest { id: number; box: ImageBox; selected: Uint8Array }
export interface FieldResponse { id: number; signed: Float32Array }

/** Builds selection distance fields off the main thread, so brushing and sketching stay smooth. */
self.onmessage = (event: MessageEvent<FieldRequest>) => {
  const { id, box, selected } = event.data;
  const { signed } = distanceField(box, selected);
  (self as unknown as Worker).postMessage({ id, signed } satisfies FieldResponse, [signed.buffer]);
};
