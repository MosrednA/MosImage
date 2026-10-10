import { DragEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { LayerRecord, fileUrl } from "../api";
import { MAX_LAYERS, variantIndex } from "../layers/stack";
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
  /** The selected layer's mask is what the tools paint. */
  masking: boolean;
  /** The selected layer shows as its mask alone. */
  maskView: boolean;
  onMask: (id: string) => void;
  onEditMask: (id: string) => void;
  onViewMask: (id: string) => void;
  onSwitchMask: (id: string) => void;
  onDeleteMask: (id: string) => void;
  /** From the picture's selection when there is one; `hideAll` starts with the whole layer hidden. */
  onAddMask: (id: string, hideAll: boolean) => void;
  /** Shows the next (1) or previous (-1) variant of an AI layer. */
  onVariant: (id: string, step: number) => void;
  onSeparateVariants: (id: string) => void;
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
        <span className={"layer-thumb" + (layer.id === activeId && !props.masking ? " targeted" : "")} title={variantIndex(layer) >= 0 ? `Variant ${variantIndex(layer) + 1} of ${layer.variant_paths.length} · , and . switch` : "Layer pixels"}>
          <img crossOrigin="anonymous" src={fileUrl(layer.image_url)} alt="" draggable={false} />
          {variantIndex(layer) >= 0 && <span className="variant-badge" aria-hidden="true">{variantIndex(layer) + 1}/{layer.variant_paths.length}</span>}
        </span>
        {layer.mask_url && <MaskThumb layer={layer} targeted={layer.id === activeId && props.masking} disabled={disabled}
          onEdit={() => props.onEditMask(layer.id)} onView={() => props.onViewMask(layer.id)} onSwitch={() => props.onSwitchMask(layer.id)} />}
        {renaming === layer.id
          ? <RenameField name={layer.name} onDone={name => { setRenaming(null); if (name && name !== layer.name) props.onUpdate(layer.id, { name }, "Rename layer"); }} />
          : <span className="layer-name" title={`${layer.name} · Double-click to rename`} onDoubleClick={() => !disabled && setRenaming(layer.id)}>{layer.name}</span>}
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

/**
 * A layer's mask beside its thumbnail, as in Photoshop: click to paint it,
 * Shift-click to switch it off, Alt-click to see it alone.
 */
function MaskThumb({ layer, targeted, disabled, onEdit, onView, onSwitch }: { layer: LayerRecord; targeted: boolean; disabled: boolean; onEdit: () => void; onView: () => void; onSwitch: () => void }) {
  return <button type="button" className={"layer-thumb mask-thumb" + (targeted ? " targeted" : "") + (layer.mask_enabled ? "" : " off")} disabled={disabled}
    aria-pressed={targeted} aria-label={`Mask of ${layer.name}${layer.mask_enabled ? "" : ", switched off"}`}
    title={`Layer mask: white shows, black hides\nClick to paint it · Shift-click switches it ${layer.mask_enabled ? "off" : "on"} · Alt-click shows it alone`}
    onClick={event => {
      event.stopPropagation();
      if (event.shiftKey) onSwitch(); else if (event.altKey) onView(); else onEdit();
    }}>
    <img crossOrigin="anonymous" src={fileUrl(layer.mask_url!)} alt="" draggable={false} />
  </button>;
}

/**
 * The proposals kept in an AI layer, like the variations of a generative
 * layer in Photoshop: pick one, step with , and ., or give each its own layer.
 */
function LayerVariants({ layer, disabled, onUpdate, onVariant, onSeparateVariants }: Pick<Props, "disabled" | "onUpdate" | "onVariant" | "onSeparateVariants"> & { layer: LayerRecord }) {
  const shown = variantIndex(layer);
  return <div className="layer-variants" role="group" aria-label="Variants">
    <div className="layer-variants-head">
      <span>Variant {shown + 1} of {layer.variant_paths.length}</span>
      <button className="icon-button" disabled={disabled} aria-label="Previous variant" title="Previous variant · ," onClick={() => onVariant(layer.id, -1)}><Icon name="chevronLeft" /></button>
      <button className="icon-button" disabled={disabled} aria-label="Next variant" title="Next variant · ." onClick={() => onVariant(layer.id, 1)}><Icon name="chevronRight" /></button>
    </div>
    <div className="layer-variant-list">
      {layer.variant_urls.map((url, index) => <button key={url} className="layer-thumb" disabled={disabled} aria-pressed={index === shown} aria-label={`Show variant ${index + 1}`}
        onClick={() => index !== shown && onUpdate(layer.id, { image_path: layer.variant_paths[index], image_url: url }, "Switch variant")}>
        <img crossOrigin="anonymous" src={fileUrl(url)} alt="" draggable={false} />
      </button>)}
      <button className="quiet-button" disabled={disabled} title="Give every variant a layer of its own; the others start hidden" onClick={() => onSeparateVariants(layer.id)}>To layers</button>
    </div>
  </div>;
}

function LayerProperties({ layer, disabled, onUpdate, onPreview, onRemove, onDuplicate, onReset, onBlend, onRemoveBackground, masking, ...props }: Props & { layer: LayerRecord }) {
  const [opacity, setOpacity] = useState(Math.round(layer.opacity * 100));
  useEffect(() => setOpacity(Math.round(layer.opacity * 100)), [layer.opacity]);
  const commitOpacity = () => { if (opacity / 100 !== layer.opacity) onUpdate(layer.id, { opacity: opacity / 100 }, "Change opacity"); };
  return <section className="layer-properties" aria-label="Layer settings">
    {variantIndex(layer) >= 0 && <LayerVariants layer={layer} disabled={disabled} onUpdate={onUpdate} onVariant={props.onVariant} onSeparateVariants={props.onSeparateVariants} />}
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
      <button className="icon-button" disabled={disabled} title="Duplicate layer" aria-label="Duplicate layer" onClick={() => onDuplicate(layer.id)}><Icon name="duplicate" /></button>
      <button className="icon-button" disabled={disabled} title="Delete layer · Delete" aria-label="Delete layer" onClick={() => onRemove(layer.id)}><Icon name="trash" /></button>
    </div>
    {layer.mask_url
      ? <div className="layer-mask-row" role="group" aria-label="Layer mask">
        <button className="layer-mask" aria-pressed={masking} disabled={disabled}
          title={masking ? "Back to the layer's pixels · Esc" : "Paint the mask: Brush shows, Eraser hides. Object, Lasso and Box work too"}
          onClick={() => props.onMask(layer.id)}><Icon name="brush" />{masking ? "Done" : "Paint mask"}</button>
        <button className="icon-button" disabled={disabled} aria-pressed={!layer.mask_enabled} aria-label={layer.mask_enabled ? "Switch the mask off" : "Switch the mask on"}
          title={(layer.mask_enabled ? "Switch the mask off" : "Switch the mask on") + " · Shift-click its thumbnail"} onClick={() => props.onSwitchMask(layer.id)}><Icon name={layer.mask_enabled ? "eye" : "eyeOff"} /></button>
        <button className="icon-button" disabled={disabled} aria-pressed={props.maskView} aria-label="Show the mask alone"
          title="Show the mask alone, in grey · Alt-click its thumbnail" onClick={() => props.onViewMask(layer.id)}><Icon name="image" /></button>
        <button className="icon-button" disabled={disabled} aria-label="Invert layer mask" title="Swap what is hidden and shown" onClick={() => props.onInvertMask(layer.id)}><Icon name="invert" /></button>
        <button className="icon-button" disabled={disabled} aria-label="Delete layer mask" title="Delete the mask and show the whole layer" onClick={() => props.onDeleteMask(layer.id)}><Icon name="trash" /></button>
      </div>
      : <button className="layer-mask" disabled={disabled}
        title={props.canCopySelection ? "Show only the part of this layer inside your selection · Alt: start with everything hidden instead" : "Add a mask that shows the whole layer, then paint parts away · Alt: start with everything hidden"}
        onClick={event => props.onAddMask(layer.id, event.altKey)}><Icon name="plus" />{props.canCopySelection ? "Mask from selection" : "Add mask"}</button>}
    <button className="layer-cutout" disabled={disabled} title="Keep the main subject with soft edges for hair and fur; the background is hidden by the layer mask"
      onClick={() => onRemoveBackground(layer.id)}><Icon name="scissors" />Remove background</button>
    <button className="layer-blend" disabled={disabled || !layer.visible} title={layer.visible ? "Let the AI match its light, colours, shadows and edges to the picture" : "Show the layer to blend it in"}
      onClick={() => onBlend(layer.id)}><Icon name="spark" />Blend in</button>
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
