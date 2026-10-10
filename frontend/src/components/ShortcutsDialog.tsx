import { ControlDialog } from "./ControlDialog";

const GROUPS: [string, [string, string][]][] = [
  ["Tools", [["W", "Object selection"], ["B", "Brush"], ["E", "Eraser"], ["L", "Lasso"], ["M", "Box"], ["P", "Sketch"], ["V / Ctrl + T", "Transform layers"], ["H", "Pan"], ["C", "Crop canvas"], ["Shift + C", "Expand canvas"]]],
  ["Selection", [
    ["Click / right-click", "Include / exclude an object point"],
    ["Click a marker", "Remove that point"],
    ["Backspace", "Remove the last point"],
    ["Alt + drag", "Subtract from the selection"],
    ["[ and ]", "Smaller / larger brush"],
    ["Q", "Show or hide the selection"],
    ["Ctrl + Shift + I", "Invert the selection"],
    ["Ctrl + D / Esc", "Deselect"],
  ]],
  ["Layers", [
    ["Ctrl + V", "Paste an image as a layer"],
    ["Ctrl + J", "Copy the selection to a layer"],
    ["Ctrl + click a layer", "Add it to the selected layers, or take it out"],
    ["Shift + click a layer", "Select the layers in between"],
    ["Ctrl + click a thumbnail", "Select that layer's pixels"],
    ["Right-click a layer", "Merge, flatten and more"],
    ["Ctrl + E", "Merge down, or merge the selected layers"],
    ["Ctrl + Shift + E", "Merge the visible layers"],
    ["Click a mask thumbnail", "Paint that layer's mask"],
    ["B / E on a layer", "Paint its mask: show / hide"],
    ["X", "Swap the mask brush between white and black"],
    ["Shift + click a mask", "Switch the mask off or on"],
    ["Alt + click a mask", "Show the mask alone, in grey"],
    ["\\", "Tint what the mask hides in red"],
    [", and .", "Previous / next variant of an AI layer"],
    ["Shift + drag a corner", "Stretch freely"],
    ["Shift + rotate", "Snap to 15°"],
    ["Arrows / Shift + arrows", "Nudge 1 px / 10 px"],
    ["Delete", "Delete the selected layers"],
    ["Esc", "Deselect the layers"],
  ]],
  ["View", [["Space + drag", "Pan with any tool"], ["Ctrl + scroll", "Zoom at the cursor"], ["Ctrl + 0", "Fit to window"], ["Ctrl + 1", "Actual pixels"]]],
  ["Edit", [
    ["Ctrl + Enter", "Generate, or apply a proposal"],
    ["Shift + Backspace", "Fill the selection by your prompt"],
    ["← →", "Previous / next proposal"],
    ["Ctrl + Z", "Undo the selection, then the image"],
    ["Ctrl + Shift + Z", "Redo"],
    ["Ctrl + O", "Open an image"],
    ["Ctrl + S", "Export the picture"],
    ["Enter / Esc", "Apply / cancel a crop"],
  ]],
];

export function ShortcutsDialog({ onClose }: { onClose: () => void }) {
  return <ControlDialog title="Keyboard shortcuts" description="Work faster without leaving the canvas. Press ? to open this list." onClose={onClose}>
    <div className="shortcut-groups">
      {GROUPS.map(([title, items]) => <section key={title}>
        <h3>{title}</h3>
        <dl>{items.map(([keys, action]) => <div key={keys}><dt><kbd>{keys}</kbd></dt><dd>{action}</dd></div>)}</dl>
      </section>)}
    </div>
  </ControlDialog>;
}
