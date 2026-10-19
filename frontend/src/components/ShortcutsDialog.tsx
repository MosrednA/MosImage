import { ControlDialog } from "./ControlDialog";
import { ALT, MOD } from "../platform";

const GROUPS: [string, [string, string][]][] = [
  ["Tools", [["W", "Object selection"], ["B", "Brush"], ["E", "Eraser"], ["L", "Lasso"], ["M", "Box"], ["P", "Sketch"], [`V / ${MOD} + T`, "Transform layers"], ["H", "Pan"], ["C", "Crop canvas"], ["Shift + C", "Expand canvas"]]],
  ["Selection", [
    ["Click / right-click", "Include / exclude an object point"],
    ["Click a marker", "Remove that point"],
    ["Backspace", "Remove the last point"],
    [`${ALT} + drag`, "Subtract from the selection"],
    ["[ and ]", "Smaller / larger brush"],
    ["Q", "Show or hide the selection"],
    [`${MOD} + Shift + I`, "Invert the selection"],
    [`${MOD} + D / Esc`, "Deselect"],
    ["Drag the action bar's grip", "Move the bar; double-click the grip to put it back"],
  ]],
  ["Layers", [
    [`${MOD} + V`, "Paste an image as a layer"],
    [`${MOD} + J`, "Copy the selection to a layer"],
    [`${MOD} + click a layer`, "Add it to the selected layers, or take it out"],
    ["Shift + click a layer", "Select the layers in between"],
    [`${MOD} + click a thumbnail`, "Select that layer's pixels"],
    ["Right-click a layer", "Merge, flatten and more"],
    [`${MOD} + E`, "Merge down, or merge the selected layers"],
    [`${MOD} + Shift + E`, "Merge the visible layers"],
    ["Click a mask thumbnail", "Paint that layer's mask"],
    ["B / E on a layer", "Paint its mask: show / hide"],
    ["X", "Swap the mask brush between white and black"],
    ["Shift + click a mask", "Switch the mask off or on"],
    [`${ALT} + click a mask`, "Show the mask alone, in grey"],
    ["\\", "Tint what the mask hides in red"],
    [", and .", "Previous / next variant of an AI layer"],
    ["Shift + drag a corner", "Stretch freely"],
    ["Shift + rotate", "Snap to 15°"],
    ["Arrows / Shift + arrows", "Nudge 1 px / 10 px"],
    ["Delete", "Delete the selected layers"],
    ["Esc", "Deselect the layers"],
  ]],
  ["View", [["Space + drag", "Pan with any tool"], [`${MOD} + scroll`, "Zoom at the cursor"], [`${MOD} + 0`, "Fit to window"], [`${MOD} + 1`, "Actual pixels"]]],
  ["Edit", [
    [`${MOD} + Enter`, "Generate, or apply a proposal"],
    ["Shift + Backspace", "Fill the selection by your prompt"],
    ["← →", "Previous / next proposal"],
    [`${MOD} + Z`, "Undo the selection, then the image"],
    [`${MOD} + Shift + Z`, "Redo"],
    [`${MOD} + O`, "Open an image"],
    [`${MOD} + S`, "Export the picture"],
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
