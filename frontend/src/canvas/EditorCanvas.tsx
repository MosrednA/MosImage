import { PointerEvent as ReactPointerEvent, useEffect, useRef, useState } from "react";
import { SelectionPoint } from "../api";
import { paddedContextBox } from "./context";
import { ImageSize, viewportPointToImage } from "./coordinates";
import { CanvasToolbar } from "./CanvasToolbar";
import { CanvasTool, toolStatus } from "./tools";
import { useMaskLayer } from "./useMaskLayer";
import { useViewport } from "./useViewport";

export type { CanvasTool } from "./tools";

interface EditorCanvasProps {
  imageUrl: string | null;
  previewImageUrl?: string | null;
  reviewing: boolean;
  imageName?: string;
  onMaskChange: (mask: Blob | null) => void;
  tool: CanvasTool;
  onToolChange: (tool: CanvasTool) => void;
  disabled: boolean;
  maskImportUrl: string | null;
  clearToken: number;
  points: SelectionPoint[];
  onObjectPoint: (point: SelectionPoint) => void;
  onImageSize: (size: ImageSize) => void;
  onError: (message: string) => void;
  contextPadding: number;
}

export function EditorCanvas({ imageUrl, previewImageUrl, reviewing, imageName, onMaskChange, tool, onToolChange, disabled, maskImportUrl, clearToken, points, onObjectPoint, onImageSize, onError, contextPadding }: EditorCanvasProps) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const imageSize = useImageSize(imageUrl, onImageSize, onError);
  const [brushSize, setBrushSize] = useState(64);
  const [edgeSize, setEdgeSize] = useState(8);
  const [showContext, setShowContext] = useState(true);
  const viewport = useViewport(stageRef, imageSize, tool === "move");
  const mask = useMaskLayer({ imageUrl, imageSize, disabled, brushSize, edgeSize, clearToken, maskImportUrl, onMaskChange, onError });
  const { zoom, displaySize } = viewport;
  const contextBox = mask.selectionBounds && imageSize ? paddedContextBox(mask.selectionBounds, imageSize, contextPadding) : null;

  function beginGesture(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (disabled || tool === "move" || event.button !== 0 || !imageSize) return;
    if (tool === "object") {
      const point = viewportPointToImage({ x: event.clientX, y: event.clientY }, event.currentTarget.getBoundingClientRect(), imageSize);
      onObjectPoint({ x: Math.min(point.x, imageSize.width - 1), y: Math.min(point.y, imageSize.height - 1), label: event.altKey ? 0 : 1 });
      return;
    }
    mask.gesture.begin(event, tool);
  }

  return (
    <section className="editor-canvas" aria-label="Image editor">
      <CanvasToolbar tool={tool} onToolChange={onToolChange} reviewing={reviewing} disabled={disabled} hasImage={!!imageSize} hasSelection={mask.hasSelection}
        showContext={showContext} onToggleContext={() => setShowContext(value => !value)}
        brushSize={brushSize} onBrushSize={setBrushSize} edgeSize={edgeSize} onEdgeSize={setEdgeSize}
        canUndo={mask.canUndo} canRedo={mask.canRedo} onUndo={mask.undo} onRedo={mask.redo} onTransform={mask.transform} onClear={mask.clear}
        zoom={zoom} fitToStage={viewport.fitToStage} atActualSize={!viewport.fitToStage && viewport.manualZoom === 1} onZoom={viewport.setZoom} onFit={viewport.fit} />

      <div ref={stageRef} className="editor-stage" data-tool={tool} {...viewport.panHandlers}>
        {imageUrl && imageSize && displaySize ? (
          <div className="editor-surface" style={{ width: displaySize.width, height: displaySize.height }}>
            <img src={previewImageUrl || imageUrl} alt={imageName || "Source image"} draggable={false} />
            <canvas
              ref={mask.canvasRef}
              width={imageSize.width}
              height={imageSize.height}
              data-tool={tool}
              aria-label="Selection canvas"
              style={{ pointerEvents: disabled ? "none" : undefined, opacity: reviewing ? 0 : undefined }}
              onPointerDown={beginGesture}
              onPointerMove={mask.gesture.update}
              onPointerUp={mask.gesture.finish}
              onPointerCancel={mask.gesture.cancel}
            />
            {!reviewing && points.map((point, index) => <i key={index} className={`selection-point ${point.label ? "positive" : "negative"}`} style={{ left: point.x * zoom, top: point.y * zoom }}>{point.label ? "+" : "−"}</i>)}
            {!reviewing && showContext && contextBox && <div className="context-boundary" aria-label="Protected context boundary" style={{ left: contextBox.x * zoom, top: contextBox.y * zoom, width: contextBox.width * zoom, height: contextBox.height * zoom }}><span>Context · stays unchanged</span></div>}
          </div>
        ) : (
          <div className="canvas-empty">
            <span className="empty-mark">+</span>
            <strong>Open, drop, or choose a recent image</strong>
            <span>Your mask stays aligned to the image at its native resolution.</span>
          </div>
        )}
      </div>

      <footer className="canvas-status">
        <span>{imageSize ? `${imageSize.width} × ${imageSize.height} px` : "No document"}</span>
        <span>{reviewing ? "Apply to save this proposal" : toolStatus(tool)}</span>
      </footer>
    </section>
  );
}

function useImageSize(imageUrl: string | null, onImageSize: (size: ImageSize) => void, onError: (message: string) => void): ImageSize | null {
  const [imageSize, setImageSize] = useState<ImageSize | null>(null);
  useEffect(() => {
    let alive = true;
    setImageSize(null);
    if (!imageUrl) return;
    const image = new Image();
    image.onload = () => {
      if (!alive) return;
      const size = { width: image.naturalWidth, height: image.naturalHeight };
      setImageSize(size);
      onImageSize(size);
    };
    image.onerror = () => { if (alive) onError("Could not open this image."); };
    image.src = imageUrl;
    return () => { alive = false; };
  }, [imageUrl, onImageSize]);
  return imageSize;
}
