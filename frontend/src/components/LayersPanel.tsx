import { DragEvent, KeyboardEvent, MouseEvent, useEffect, useRef, useState } from "react";
import { LayerRecord, fileUrl } from "../api";
import { MAX_LAYERS, MergeKind, replacementOf, variantIndex } from "../layers/stack";
import { ContextMenu, MenuEntry } from "./ContextMenu";
import { Icon } from "./Icon";
import { ALT, MOD } from "../platform";

interface Props {
  layers: LayerRecord[];
  activeId: string | null;
  /** Every selected layer, the active one among them. */
  selectedIds: string[];
  uploading: number;
  disabled: boolean;
  canCopySelection: boolean;
  /** Which merges have something to merge now. */
  mergeable: Record<MergeKind, boolean>;
  onSelect: (id: string | null) => void;
  /** Ctrl-click: adds the layer to the selection or takes it out. */
  onToggleSelect: (id: string) => void;
  /** Shift-click: selects the layers from the last one clicked. */
  onSelectRange: (id: string) => void;
  /** Ctrl-click on the thumbnail: selects the layer's pixels on the picture. */
  onSelectPixels: (id: string) => void;
  onMerge: (kind: MergeKind) => void;
  onAdd: (files: File[]) => void;
  onCopySelection: () => void;
  onUpdate: (id: string, patch: Partial<LayerRecord>, label: string) => void;
  onPreview: (id: string, patch: Partial<LayerRecord>) => void;
  onRemove: (ids: string[]) => void;
  onSetVisible: (ids: string[], visible: boolean) => void;
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
  const { layers, activeId, selectedIds, disabled, mergeable } = props;
  const picker = useRef<HTMLInputElement>(null);
  const [dropping, setDropping] = useState(false);
  const [dragTarget, setDragTarget] = useState<number | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);
  const active = layers.find(layer => layer.id === activeId) ?? null;
  const selected = layers.filter(layer => selectedIds.includes(layer.id));
  const full = layers.length >= MAX_LAYERS;
  // Top of the stack first, like any layers panel.
  const listed = [...layers].reverse();

  function click(event: MouseEvent, id: string) {
    if (event.shiftKey) props.onSelectRange(id);
    else if (event.ctrlKey || event.metaKey) props.onToggleSelect(id);
    else props.onSelect(id);
  }
  /** Right-click works on the selection, or on the layer alone when it was not selected, as in Photoshop. */
  function openMenu(event: MouseEvent, id: string | null) {
    // Renaming keeps the text field's own menu.
    if ((event.target as HTMLElement).closest("input")) return;
    event.preventDefault(); event.stopPropagation();
    if (disabled) return;
    if (id && !selectedIds.includes(id)) props.onSelect(id);
    setMenu({ x: event.clientX, y: event.clientY });
  }
  function openMenuAt(element: HTMLElement) {
    const box = element.getBoundingClientRect();
    setMenu({ x: box.left, y: box.bottom + 2 });
  }

  function menuItems(): MenuEntry[] {
    const merges: MenuEntry[] = [
      { label: "Merge visible", shortcut: `${MOD}+Shift+E`, disabled: !mergeable.visible, title: "Merge every visible layer into one; hidden layers stay", onSelect: () => props.onMerge("visible") },
      { label: "Flatten image", disabled: !mergeable.flatten, title: "Merge everything into one Background layer; hidden layers are discarded", onSelect: () => props.onMerge("flatten") },
    ];
    if (selected.length > 1) {
      const anyHidden = selected.some(layer => !layer.visible);
      return [
        { label: "Merge layers", shortcut: `${MOD}+E`, disabled: !mergeable.selected, onSelect: () => props.onMerge("selected") },
        { label: anyHidden ? "Show layers" : "Hide layers", onSelect: () => props.onSetVisible(selectedIds, anyHidden) },
        { label: `Delete ${selected.length} layers`, shortcut: "Delete", danger: true, onSelect: () => props.onRemove(selectedIds) },
        "separator", ...merges,
      ];
    }
    const layer = selected[0] ?? null;
    if (!layer) return [
      { label: "Add image…", disabled: full, onSelect: () => picker.current?.click() },
      { label: "Selection to layer", shortcut: `${MOD}+J`, disabled: full || !props.canCopySelection, onSelect: props.onCopySelection },
      "separator", ...merges,
    ];
    return [
      { label: "Select pixels", shortcut: `${MOD}+click thumbnail`, onSelect: () => props.onSelectPixels(layer.id) },
      { label: layer.visible ? "Hide layer" : "Show layer", onSelect: () => props.onSetVisible([layer.id], !layer.visible) },
      { label: "Rename", onSelect: () => setRenaming(layer.id) },
      { label: "Duplicate layer", disabled: full, onSelect: () => props.onDuplicate(layer.id) },
      { label: "Delete layer", shortcut: "Delete", danger: true, onSelect: () => props.onRemove([layer.id]) },
      "separator",
      { label: "Merge down", shortcut: `${MOD}+E`, disabled: !mergeable.down, title: "Merge onto the next visible layer below", onSelect: () => props.onMerge("down") },
      ...merges,
    ];
  }

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
      <button disabled={disabled || full || !props.canCopySelection} title={`Copy the selected area to a new layer · ${MOD} + J`} onClick={props.onCopySelection}><Icon name="scissors" />Selection</button>
      <button className="icon-button" disabled={disabled} aria-haspopup="menu" aria-expanded={!!menu} aria-label="Layer menu" title="Merge, flatten and more · or right-click a layer"
        onClick={event => openMenuAt(event.currentTarget)}><Icon name="more" /></button>
    </div>
    <div className="layer-list" role="listbox" aria-label="Layers" aria-multiselectable="true" onContextMenu={event => openMenu(event, null)}>
      {!layers.length && !props.uploading && <button className="reference-empty layers-empty" disabled={disabled} onClick={() => picker.current?.click()}>
        <Icon name="layers" />
        <strong>An empty canvas</strong>
        <span>Add a photo or a transparent PNG, or paste one. Move, scale, stretch and rotate it, and mask parts away.</span>
        <small>Drop or paste images, or click to browse</small>
      </button>}
      {Array.from({ length: props.uploading }, (_, index) => <div key={"upload-" + index} className="layer-row uploading"><span className="reference-spinner" aria-label="Working on layers" /></div>)}
      {listed.map((layer, index) => {
        const isSelected = selectedIds.includes(layer.id);
        const replacement = replacementOf(layers, layer.id);
        return <div key={layer.id} role="option" aria-selected={isSelected} tabIndex={0}
          className={"layer-row" + (isSelected ? " active" : "") + (layer.visible ? "" : " hidden") + (dragTarget === index ? " drop-target" : "")}
          draggable={!disabled && renaming !== layer.id}
          onClick={event => click(event, layer.id)}
          onContextMenu={event => openMenu(event, layer.id)}
          onKeyDown={event => {
            if (event.target !== event.currentTarget) return;
            if (event.key === "Enter") props.onSelect(layer.id);
            if (event.key === "ContextMenu" || (event.shiftKey && event.key === "F10")) {
              event.preventDefault();
              if (!isSelected) props.onSelect(layer.id);
              openMenuAt(event.currentTarget);
            }
          }}
          onDragStart={event => { event.dataTransfer.setData(LAYER_DRAG_TYPE, layer.id); event.dataTransfer.effectAllowed = "move"; }}
          onDragOver={event => { if (event.dataTransfer.types.includes(LAYER_DRAG_TYPE)) { event.preventDefault(); event.stopPropagation(); setDragTarget(index); } }}
          onDrop={event => drop(event, index)}
          onDragEnd={() => setDragTarget(null)}>
          <button className="icon-button" disabled={disabled} aria-pressed={!layer.visible} title={layer.visible ? "Hide layer" : "Show layer"} aria-label={layer.visible ? `Hide ${layer.name}` : `Show ${layer.name}`}
            onClick={event => { event.stopPropagation(); props.onUpdate(layer.id, { visible: !layer.visible }, layer.visible ? "Hide layer" : "Show layer"); }}>
            <Icon name={layer.visible ? "eye" : "eyeOff"} />
          </button>
          <span className={"layer-thumb" + (layer.id === activeId && !props.masking ? " targeted" : "")}
            title={(variantIndex(layer) >= 0 ? `Variant ${variantIndex(layer) + 1} of ${layer.variant_paths.length} · , and . switch` : "Layer pixels") + `\n${MOD}-click to select them`}
            onClick={event => { if (!(event.ctrlKey || event.metaKey) || disabled) return; event.stopPropagation(); props.onSelectPixels(layer.id); }}>
            <img crossOrigin="anonymous" src={fileUrl(layer.image_url)} alt="" draggable={false} />
            {variantIndex(layer) >= 0 && <span className="variant-badge" aria-hidden="true">{variantIndex(layer) + 1}/{layer.variant_paths.length}</span>}
          </span>
          {layer.mask_url && <MaskThumb layer={layer} targeted={layer.id === activeId && props.masking} disabled={disabled}
            onEdit={() => props.onEditMask(layer.id)} onView={() => props.onViewMask(layer.id)} onSwitch={() => props.onSwitchMask(layer.id)} />}
          {renaming === layer.id
            ? <RenameField name={layer.name} onDone={name => { setRenaming(null); if (name && name !== layer.name) props.onUpdate(layer.id, { name }, "Rename layer"); }} />
            : <span className="layer-name" onDoubleClick={() => !disabled && setRenaming(layer.id)}
              title={`${layer.name} · Double-click to rename` + (replacement ? `\nKept hidden under ${replacement.name}, which replaced it. Deleting ${replacement.name} shows it again; merging it lets this one go.` : "")}>
              <span>{layer.name}</span>
              {replacement && <small>Original of {replacement.name}</small>}
            </span>}
        </div>;
      })}
    </div>
    {selected.length > 1
      ? <SelectedLayers {...props} selected={selected} />
      : active && <LayerProperties key={active.id} {...props} layer={active} />}
    {dropping && <div className="reference-drop">Drop to add layers</div>}
    {menu && <ContextMenu x={menu.x} y={menu.y} label="Layer actions" items={menuItems()} onClose={() => setMenu(null)} />}
  </div>;
}

/** What works on several layers at once. */
function SelectedLayers({ selected, selectedIds, disabled, mergeable, onMerge, onSetVisible, onRemove }: Props & { selected: LayerRecord[] }) {
  const anyHidden = selected.some(layer => !layer.visible);
  return <section className="layer-properties" aria-label="Selected layers">
    <div className="layer-group" role="group" aria-labelledby="layer-group-selected">
      <h3 className="layer-group-title" id="layer-group-selected"><Icon name="layers" />{selected.length} layers</h3>
      <div className="layer-buttons">
        <button disabled={disabled || !mergeable.selected} title={`Merge the visible selected layers into one · ${MOD} + E`} onClick={() => onMerge("selected")}><Icon name="merge" />Merge</button>
        <button disabled={disabled} title={anyHidden ? "Show the selected layers" : "Hide the selected layers"} onClick={() => onSetVisible(selectedIds, anyHidden)}>
          <Icon name={anyHidden ? "eye" : "eyeOff"} />{anyHidden ? "Show" : "Hide"}</button>
        <button disabled={disabled} title="Delete the selected layers · Delete" onClick={() => onRemove(selectedIds)}><Icon name="trash" />Delete</button>
      </div>
      <small className="layer-hint">{MOD}-click adds or removes a layer · Shift-click selects a range</small>
    </div>
  </section>;
}

/**
 * A layer's mask beside its thumbnail, as in Photoshop: click to paint it,
 * Shift-click to switch it off, Alt-click to see it alone.
 */
function MaskThumb({ layer, targeted, disabled, onEdit, onView, onSwitch }: { layer: LayerRecord; targeted: boolean; disabled: boolean; onEdit: () => void; onView: () => void; onSwitch: () => void }) {
  return <button type="button" className={"layer-thumb mask-thumb" + (targeted ? " targeted" : "") + (layer.mask_enabled ? "" : " off")} disabled={disabled}
    aria-pressed={targeted} aria-label={`Mask of ${layer.name}${layer.mask_enabled ? "" : ", switched off"}`}
    title={`Layer mask: white shows, black hides\nClick to paint it · Shift-click switches it ${layer.mask_enabled ? "off" : "on"} · ${ALT}-click shows it alone`}
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

function LayerProperties({ layer, disabled, onUpdate, onPreview, onRemove, onDuplicate, onReset, onBlend, onRemoveBackground, masking, mergeable, ...props }: Props & { layer: LayerRecord }) {
  const [opacity, setOpacity] = useState(Math.round(layer.opacity * 100));
  useEffect(() => setOpacity(Math.round(layer.opacity * 100)), [layer.opacity]);
  const commitOpacity = () => { if (opacity / 100 !== layer.opacity) onUpdate(layer.id, { opacity: opacity / 100 }, "Change opacity"); };
  // Two groups, as in Photoshop's Properties panel: what changes the layer's pixels and placement, and what changes its mask.
  return <section className="layer-properties" aria-label="Layer settings">
    {variantIndex(layer) >= 0 && <LayerVariants layer={layer} disabled={disabled} onUpdate={onUpdate} onVariant={props.onVariant} onSeparateVariants={props.onSeparateVariants} />}
    <div className="layer-group" role="group" aria-labelledby="layer-group-pixels">
      <h3 className="layer-group-title" id="layer-group-pixels"><Icon name="image" />Layer</h3>
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
        <button className="icon-button" disabled={disabled} title="Delete layer · Delete" aria-label="Delete layer" onClick={() => onRemove([layer.id])}><Icon name="trash" /></button>
      </div>
      <div className="layer-buttons">
        <button disabled={disabled} title={`Select where this layer shows, to fill, mask or copy it · ${MOD}-click its thumbnail`} onClick={() => props.onSelectPixels(layer.id)}>
          <Icon name="rectangle" />Select pixels</button>
        <button disabled={disabled || !mergeable.down} title={mergeable.down ? `Merge onto the next visible layer below · ${MOD} + E` : "Needs a visible layer below, and this one shown"}
          onClick={() => props.onMerge("down")}><Icon name="merge" />Merge down</button>
      </div>
      <button className="layer-blend" disabled={disabled || !layer.visible} title={layer.visible ? "Let the AI match its light, colours, shadows and edges to the picture" : "Show the layer to blend it in"}
        onClick={() => onBlend(layer.id)}><Icon name="spark" />Blend in</button>
    </div>
    <div className={"layer-group mask-group" + (masking ? " masking" : "")} role="group" aria-labelledby="layer-group-mask">
      <h3 className="layer-group-title" id="layer-group-mask"><Icon name="invert" />Mask
        <small>{!layer.mask_url ? "none: the whole layer shows" : layer.mask_enabled ? "white shows · black hides" : "switched off"}</small></h3>
      {layer.mask_url
        ? <>
          <button className="layer-mask" aria-pressed={masking} disabled={disabled}
            title={masking ? "Back to the layer's pixels · Esc" : "Paint the mask from white to black: Brush paints its tone, Eraser hides"}
            onClick={() => props.onMask(layer.id)}><Icon name="brush" />{masking ? "Done painting" : "Paint mask"}</button>
          <div className="mask-tools">
            <button className="mask-tool" disabled={disabled} aria-pressed={!layer.mask_enabled}
              title={(layer.mask_enabled ? "Switch the mask off and show the whole layer" : "Switch the mask on") + " · Shift-click its thumbnail"} onClick={() => props.onSwitchMask(layer.id)}>
              <Icon name={layer.mask_enabled ? "eye" : "eyeOff"} /><span>{layer.mask_enabled ? "Turn off" : "Turn on"}</span></button>
            <button className="mask-tool" disabled={disabled} aria-pressed={props.maskView}
              title={`Show the mask alone, in grey · ${ALT}-click its thumbnail`} onClick={() => props.onViewMask(layer.id)}><Icon name="image" /><span>View</span></button>
            <button className="mask-tool" disabled={disabled} title="Swap what is hidden and shown" onClick={() => props.onInvertMask(layer.id)}><Icon name="invert" /><span>Invert</span></button>
            <button className="mask-tool" disabled={disabled} title="Delete the mask and show the whole layer" onClick={() => props.onDeleteMask(layer.id)}><Icon name="trash" /><span>Delete</span></button>
          </div>
        </>
        : <button className="layer-mask" disabled={disabled}
          title={props.canCopySelection ? `Show only the part of this layer inside your selection · ${ALT}: start with everything hidden instead` : `Add a mask that shows the whole layer, then paint parts away · ${ALT}: start with everything hidden`}
          onClick={event => props.onAddMask(layer.id, event.altKey)}><Icon name="plus" />{props.canCopySelection ? "Mask from selection" : "Add mask"}</button>}
      <button className="layer-cutout" disabled={disabled} title="Keep the main subject with soft edges for hair and fur: the mask hides the background, and you can paint it after"
        onClick={() => onRemoveBackground(layer.id)}><Icon name="scissors" />Remove background</button>
    </div>
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
