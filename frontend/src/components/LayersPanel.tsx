import { DragEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { LayerRecord, fileUrl } from "../api";
import { MAX_LAYERS } from "../layers/stack";
import { Icon } from "./Icon";

interface Props {
  layers: LayerRecord[];
  activeId: string | null;
  backgroundUrl: string | null;
  uploading: number;
  disabled: boolean;
  canCopySelection: boolean;
  onSelect: (id: string | null) => void;
  onAdd: (files: File[]) => void;
  onCopySelection: () => void;
  onUpdate: (id: string, patch: Partial<LayerRecord>, label: string) => void;
  onPreview: (id: string, patch: Partial<LayerRecord>) => void;
  onRemove: (id: string) => void;
  onDuplicate: (id: string) => void;
  onReorder: (id: string, index: number) => void;
  onReset: (id: string) => void;
  onInvertMask: (id: string) => void;
  onBlend: (id: string) => void;
  onRemoveBackground: (id: string) => void;
  masking: boolean;
  onMask: (id: string) => void;
}

const LAYER_DRAG_TYPE = "application/x-mosimage-layer";

export function LayersPanel(props: Props) {
  const { layers, activeId, disabled } = props;
  const picker = useRef<HTMLInputElement>(null);
  const [dropping, setDropping] = useState(false);
  const [dragTarget, setDragTarget] = useState<number | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const active = layers.find(layer => layer.id === activeId) ?? null;
  const full = layers.length >= MAX_LAYERS;
  // Top of the stack first, like any layers panel.
  const listed = [...layers].reverse();

  function drop(event: DragEvent<HTMLElement>, index: number | null) {
    event.preventDefault(); event.stopPropagation();
    setDropping(false); setDragTarget(null);
    if (disabled) return;
    const id = event.dataTransfer.getData(LAYER_DRAG_TYPE);
    if (id && index !== null) { props.onReorder(id, layers.length - 1 - index); return; }
    const files = Array.from(event.dataTransfer.files);
    if (files.length) props.onAdd(files);
  }
  const panelDrag = {
    onDragOver: (event: DragEvent<HTMLElement>) => {
      event.preventDefault(); event.stopPropagation();
      if (!disabled && event.dataTransfer.types.includes("Files")) setDropping(true);
    },
    onDragLeave: (event: DragEvent<HTMLElement>) => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) { setDropping(false); setDragTarget(null); } },
    onDrop: (event: DragEvent<HTMLElement>) => drop(event, null),
  };

  return <div className={"layers-panel" + (dropping ? " dropping" : "")} {...panelDrag}>
    <input ref={picker} className="visually-hidden" type="file" multiple accept="image/png,image/jpeg,image/webp"
      onChange={event => { const files = Array.from(event.target.files ?? []); if (files.length) props.onAdd(files); event.target.value = ""; }} />
    <div className="layers-actions">
      <button disabled={disabled || full} title={full ? `Up to ${MAX_LAYERS} layers` : "Add images as layers"} onClick={() => picker.current?.click()}><Icon name="plus" />Add image</button>
      <button disabled={disabled || full || !props.canCopySelection} title="Copy the selected area to a new layer · Ctrl + J" onClick={props.onCopySelection}><Icon name="scissors" />Selection</button>
    </div>
    <div className="layer-list" role="listbox" aria-label="Layers">
      {!layers.length && !props.uploading && <button className="reference-empty layers-empty" disabled={disabled} onClick={() => picker.current?.click()}>
        <Icon name="layers" />
        <strong>Stack images on your picture</strong>
        <span>Add a photo or a transparent PNG, or copy a selection. Move, scale, stretch and rotate it, and cut parts away.</span>
        <small>Drop or paste images, or click to browse</small>
      </button>}
      {Array.from({ length: props.uploading }, (_, index) => <div key={"upload-" + index} className="layer-row uploading"><span className="reference-spinner" aria-label="Adding layer" /></div>)}
      {listed.map((layer, index) => <div key={layer.id} role="option" aria-selected={layer.id === activeId} tabIndex={0}
        className={"layer-row" + (layer.id === activeId ? " active" : "") + (layer.visible ? "" : " hidden") + (dragTarget === index ? " drop-target" : "")}
        draggable={!disabled && renaming !== layer.id}
        onClick={() => props.onSelect(layer.id)}
        onKeyDown={event => { if (event.key === "Enter" && event.target === event.currentTarget) props.onSelect(layer.id); }}
        onDragStart={event => { event.dataTransfer.setData(LAYER_DRAG_TYPE, layer.id); event.dataTransfer.effectAllowed = "move"; }}
        onDragOver={event => { if (event.dataTransfer.types.includes(LAYER_DRAG_TYPE)) { event.preventDefault(); event.stopPropagation(); setDragTarget(index); } }}
        onDrop={event => drop(event, index)}
        onDragEnd={() => setDragTarget(null)}>
        <span className="layer-thumb"><img crossOrigin="anonymous" src={fileUrl(layer.image_url)} alt="" draggable={false} /></span>
        {renaming === layer.id
          ? <RenameField name={layer.name} onDone={name => { setRenaming(null); if (name && name !== layer.name) props.onUpdate(layer.id, { name }, "Rename layer"); }} />
          : <span className="layer-name" title={`${layer.name} · Double-click to rename`} onDoubleClick={() => !disabled && setRenaming(layer.id)}>{layer.name}{layer.mask_url && <small> · cut</small>}</span>}
        <button className="icon-button" disabled={disabled} aria-pressed={!layer.visible} title={layer.visible ? "Hide layer" : "Show layer"} aria-label={layer.visible ? `Hide ${layer.name}` : `Show ${layer.name}`}
          onClick={event => { event.stopPropagation(); props.onUpdate(layer.id, { visible: !layer.visible }, layer.visible ? "Hide layer" : "Show layer"); }}>
          <Icon name={layer.visible ? "eye" : "eyeOff"} />
        </button>
      </div>)}
      <div role="option" aria-selected={activeId === null} tabIndex={0} className={"layer-row background" + (activeId === null ? " active" : "")}
        onClick={() => props.onSelect(null)} onKeyDown={event => { if (event.key === "Enter") props.onSelect(null); }}>
        <span className="layer-thumb">{props.backgroundUrl && <img src={props.backgroundUrl} alt="" draggable={false} />}</span>
        <span className="layer-name">Background<small>AI results arrive as layers</small></span>
      </div>
    </div>
    {active && <LayerProperties key={active.id} {...props} layer={active} />}
    {dropping && <div className="reference-drop">Drop to add layers</div>}
  </div>;
}

function LayerProperties({ layer, disabled, onUpdate, onPreview, onRemove, onDuplicate, onReset, onInvertMask, onBlend, onRemoveBackground, masking, onMask }: Props & { layer: LayerRecord }) {
  const [opacity, setOpacity] = useState(Math.round(layer.opacity * 100));
  useEffect(() => setOpacity(Math.round(layer.opacity * 100)), [layer.opacity]);
  const commitOpacity = () => { if (opacity / 100 !== layer.opacity) onUpdate(layer.id, { opacity: opacity / 100 }, "Change opacity"); };
  return <section className="layer-properties" aria-label="Layer settings">
    <label className="layer-opacity">
      <span>Opacity</span>
      <input type="range" min={0} max={100} value={opacity} disabled={disabled}
        onChange={event => { const value = Number(event.target.value); setOpacity(value); onPreview(layer.id, { opacity: value / 100 }); }}
        onPointerUp={commitOpacity} onKeyUp={commitOpacity} onBlur={commitOpacity} />
      <output>{opacity}%</output>
    </label>
    <div className="layer-tools">
      <button className="icon-button" disabled={disabled} title="Flip horizontally" aria-label="Flip horizontally" onClick={() => onUpdate(layer.id, { scale_x: -layer.scale_x }, "Flip layer")}><Icon name="flipH" /></button>
      <button className="icon-button" disabled={disabled} title="Flip vertically" aria-label="Flip vertically" onClick={() => onUpdate(layer.id, { scale_y: -layer.scale_y }, "Flip layer")}><Icon name="flipV" /></button>
      <button className="icon-button" disabled={disabled} title="Rotate 90°" aria-label="Rotate 90 degrees" onClick={() => onUpdate(layer.id, { rotation: turnedQuarter(layer.rotation) }, "Rotate layer")}><Icon name="rotate" /></button>
      <button className="icon-button" disabled={disabled} title="Reset position, size and rotation" aria-label="Reset transform" onClick={() => onReset(layer.id)}><Icon name="reset" /></button>
      <button className="icon-button" disabled={disabled} title="Swap what is hidden and shown" aria-label="Invert layer mask" onClick={() => onInvertMask(layer.id)}><Icon name="invert" /></button>
      <button className="icon-button" disabled={disabled} title="Duplicate layer" aria-label="Duplicate layer" onClick={() => onDuplicate(layer.id)}><Icon name="duplicate" /></button>
      <button className="icon-button" disabled={disabled} title="Delete layer · Delete" aria-label="Delete layer" onClick={() => onRemove(layer.id)}><Icon name="trash" /></button>
    </div>
    <button className="layer-mask" aria-pressed={masking} disabled={disabled}
      title={masking ? "Stop masking · Esc" : "Hide or restore parts of this layer with Object, Brush, Eraser, Lasso and Box"}
      onClick={() => onMask(layer.id)}><Icon name="eraser" />{masking ? "Done masking" : "Mask layer"}</button>
    <button className="layer-cutout" disabled={disabled} title="Keep the main subject with soft edges for hair and fur; the background is hidden by the layer mask"
      onClick={() => onRemoveBackground(layer.id)}><Icon name="scissors" />Remove background</button>
    <button className="layer-blend" disabled={disabled || !layer.visible} title={layer.visible ? "Let the AI match its light, colours, shadows and edges to the picture" : "Show the layer to blend it in"}
      onClick={() => onBlend(layer.id)}><Icon name="spark" />Blend in</button>
    {layer.mask_url && <button className="layer-unmask" disabled={disabled} onClick={() => onUpdate(layer.id, { mask_path: null, mask_url: null }, "Show whole layer")}>Show the whole layer again</button>}
  </section>;
}

function RenameField({ name, onDone }: { name: string; onDone: (name: string) => void }) {
  const [value, setValue] = useState(name);
  const finish = () => onDone(value.trim().slice(0, 80));
  return <input className="layer-rename" autoFocus value={value} maxLength={80} aria-label="Layer name"
    onClick={event => event.stopPropagation()} onChange={event => setValue(event.target.value)} onBlur={finish}
    onKeyDown={(event: KeyboardEvent<HTMLInputElement>) => { if (event.key === "Enter") finish(); if (event.key === "Escape") onDone(name); }} />;
}

function turnedQuarter(rotation: number): number {
  const next = Math.round((rotation + 90) / 90) * 90;
  return ((next + 180) % 360 + 360) % 360 - 180;
}
