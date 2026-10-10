import { ControlDialog } from "./ControlDialog";

const GROUPS: [string, [string, string][]][] = [
  ["Tools", [["O", "Object selection"], ["B", "Brush"], ["E", "Eraser"], ["L", "Lasso"], ["M", "Box"], ["P", "Sketch"], ["T", "Transform layers"], ["V", "Pan"], ["C", "Expand canvas"]]],
  ["Selection", [
    ["Click / right-click", "Include / exclude an object point"],
    ["Click a marker", "Remove that point"],
    ["Backspace", "Remove the last point"],
    ["Alt + drag", "Subtract from the selection"],
    ["[ and ]", "Smaller / larger brush"],
    ["Q", "Show or hide the selection"],
    ["Ctrl + I", "Invert the selection"],
    ["Esc", "Clear the selection"],
  ]],
  ["Layers", [
    ["Ctrl + V", "Paste an image as a layer"],
    ["Ctrl + J", "Copy the selection to a layer"],
    ["Click a mask thumbnail", "Paint that layer's mask"],
    ["Shift + click a mask", "Switch the mask off or on"],
    ["Alt + click a mask", "Show the mask alone, in grey"],
    ["\\", "Tint what the mask hides in red"],
    ["Shift + drag a corner", "Stretch freely"],
    ["Shift + rotate", "Snap to 15°"],
    ["Arrows / Shift + arrows", "Nudge 1 px / 10 px"],
    ["Delete", "Delete the layer"],
    ["Esc", "Back to the background"],
  ]],
  ["View", [["Space + drag", "Pan with any tool"], ["Ctrl + scroll", "Zoom at the cursor"], ["Ctrl + 0", "Fit to window"], ["Ctrl + 1", "Actual pixels"]]],
  ["Edit", [
    ["Ctrl + Enter", "Generate, or apply a proposal"],
    ["← →", "Previous / next proposal"],
    ["Ctrl + Z", "Undo the selection, then the image"],
    ["Ctrl + Shift + Z", "Redo"],
    ["Ctrl + O", "Open an image"],
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
