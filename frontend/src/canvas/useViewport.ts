import { PointerEvent as ReactPointerEvent, RefObject, useLayoutEffect, useRef, useState } from "react";
import { ImageSize } from "./coordinates";

interface PanGesture { pointerId: number; clientX: number; clientY: number; scrollLeft: number; scrollTop: number }

const STAGE_MARGIN = 40;
export const MIN_ZOOM = 0.1;
export const MAX_ZOOM = 2;

export function useViewport(stageRef: RefObject<HTMLDivElement | null>, imageSize: ImageSize | null, panEnabled: boolean) {
  const panRef = useRef<PanGesture | null>(null);
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 });
  const [fitToStage, setFitToStage] = useState(true);
  const [manualZoom, setManualZoom] = useState(1);

  useLayoutEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const updateSize = () => setStageSize({ width: stage.clientWidth, height: stage.clientHeight });
    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(stage);
    return () => observer.disconnect();
  }, []);

  // Every newly opened image starts fitted to the stage.
  useLayoutEffect(() => { if (imageSize) setFitToStage(true); }, [imageSize]);

  const fitZoom = imageSize
    ? Math.min(
        Math.max((stageSize.width - STAGE_MARGIN) / imageSize.width, 0.05),
        Math.max((stageSize.height - STAGE_MARGIN) / imageSize.height, 0.05),
        1,
      )
    : 1;
  const zoom = fitToStage ? fitZoom : manualZoom;
  const displaySize = imageSize
    ? { width: Math.round(imageSize.width * zoom), height: Math.round(imageSize.height * zoom) }
    : null;

  function setZoom(nextZoom: number) {
    setFitToStage(false);
    setManualZoom(Math.min(Math.max(nextZoom, MIN_ZOOM), MAX_ZOOM));
  }

  const panHandlers = {
    onPointerDown(event: ReactPointerEvent<HTMLDivElement>) {
      if (!panEnabled && event.button !== 1) return;
      const stage = stageRef.current;
      if (!stage) return;
      event.preventDefault();
      panRef.current = { pointerId: event.pointerId, clientX: event.clientX, clientY: event.clientY, scrollLeft: stage.scrollLeft, scrollTop: stage.scrollTop };
      stage.setPointerCapture(event.pointerId);
    },
    onPointerMove(event: ReactPointerEvent<HTMLDivElement>) {
      const pan = panRef.current;
      const stage = stageRef.current;
      if (!pan || !stage || pan.pointerId !== event.pointerId) return;
      stage.scrollLeft = pan.scrollLeft - (event.clientX - pan.clientX);
      stage.scrollTop = pan.scrollTop - (event.clientY - pan.clientY);
    },
    onPointerUp: finishPan,
    onPointerCancel: finishPan,
  };

  function finishPan(event: ReactPointerEvent<HTMLDivElement>) {
    const pan = panRef.current;
    const stage = stageRef.current;
    if (!pan || !stage || pan.pointerId !== event.pointerId) return;
    panRef.current = null;
    if (stage.hasPointerCapture(event.pointerId)) stage.releasePointerCapture(event.pointerId);
  }

  return { zoom, displaySize, fitToStage, manualZoom, fit: () => setFitToStage(true), setZoom, panHandlers };
}
