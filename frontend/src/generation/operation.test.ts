import { describe, expect, it } from "vitest";
import { ENHANCE_DETAIL, enhanceHint, facesHint, isWholeImage, joinNames, layerEditHint, operationHint, promptPlaceholder, sketchHint, submitLabel, taskFor } from "./operation";

describe("sketch, faces and enhance", () => {
  it("run on FLUX.2 Klein's edit pipeline", () => {
    expect(taskFor("sketch")).toBe("edit");
    expect(taskFor("faces")).toBe("edit");
    expect(taskFor("enhance")).toBe("edit");
    expect(isWholeImage("sketch") || isWholeImage("faces") || isWholeImage("enhance")).toBe(false);
    expect(submitLabel("enhance")).toBe("Enhance");
    expect(submitLabel("sketch")).toBe("Render");
    expect(submitLabel("faces")).toBe("Fix faces");
  });
  it("explain what happens", () => {
    expect(sketchHint("strokes", false)).toContain("Paint rough shapes");
    expect(sketchHint("strokes", true)).toContain("the rest of the picture stays");
    expect(sketchHint("whole", true)).toContain("redraws the whole picture");
    expect(sketchHint("strokes", false, ["Anna"])).toBe("Sketch where and how Anna should go, then say who is where. The sketch sets position, size and pose; the reference sets the look and colours.");
    expect(sketchHint("whole", true, ["Anna", "Tom"])).toContain("Anna and Tom keep the look and colours of their reference.");
    expect(promptPlaceholder("sketch", true, ["Anna"])).toBe("What did you draw? e.g. Anna sitting on the bench");
    expect(promptPlaceholder("sketch", true, ["Anna", "Tom"])).toBe("What did you draw? e.g. Anna and Tom sitting on the bench, Anna on the left");
    expect(facesHint(false)).toBe("Every face in the picture is sharpened from the face itself and pasted back. Expression and lighting stay the same. Apply adds the faces as a new layer.");
    expect(facesHint(false, [], "redraw")).toContain("The expression can change.");
    expect(facesHint(true, [], "faithful", { found: 3, picked: 2 })).toBe("The 2 picked faces are sharpened from the face itself and pasted back. Expression and lighting stay the same. Click a face to leave it out or add it. Apply adds the faces as a new layer.");
    expect(facesHint(false, [], "faithful", { found: 3, picked: 1 })).toMatch(/^The picked face is sharpened/);
    expect(facesHint(false, [], "faithful", { found: 3, picked: 0 })).toBe("Click a face on the picture to fix it.");
    expect(facesHint(false, [], "faithful", { found: 0, picked: 0 })).toBe("No faces found in the picture.");
    expect(facesHint(true, ["Anna"], "redraw")).toBe("Faces inside your selection are drawn anew and pasted back, for faces too blurry to sharpen. Anna keeps the look of their reference. Apply adds the faces as a new layer.");
    expect(enhanceHint(true)).toContain("to your selection;");
    expect(enhanceHint(false)).toContain("to the whole picture;");
    expect(promptPlaceholder("enhance", true)).toContain("Optional");
    // Subtle is the scale the A/B test found most natural; strong starts to reinvent.
    expect(ENHANCE_DETAIL).toEqual({ subtle: 0.4, strong: 0.7 });
    expect(promptPlaceholder("faces", true)).toContain("Optional");
  });
});

describe("reference guidance", () => {
  it("explains generation layers without asking for a selection", () => {
    expect(promptPlaceholder("generate", true)).toBe("Describe a new image to add as a layer…");
    expect(promptPlaceholder("generate", true, ["Anna"])).toBe("Where is Anna? e.g. riding a roller coaster together");
    expect(operationHint("generate", true, false, "object")).toContain("new layer");
    expect(operationHint("generate", false, false, "object")).toContain("Describe a new image");
  });
  it("edits the whole picture, with or without references", () => {
    expect(taskFor("edit")).toBe("edit");
    expect(submitLabel("edit")).toBe("Edit");
    expect(isWholeImage("edit")).toBe(true);
    expect(isWholeImage("replace")).toBe(false);
    expect(promptPlaceholder("edit", true)).toBe("How should the picture change? e.g. make it a snowy night");
    expect(promptPlaceholder("edit", true, ["Anna"])).toContain("put Anna next to the fountain");
    expect(operationHint("edit", true, false, "object")).toContain("Faces and everything you don't mention stay as they are.");
    expect(operationHint("edit", true, true, "object", ["Anna", "Tom"])).toContain("Anna and Tom keep the look of their reference.");
    // A freshly opened picture starts in Edit with the Object tool, so the hint points at selecting.
    expect(operationHint("edit", true, false, "object")).toContain("To change one part, select it first.");
    expect(operationHint("edit", true, false, "transform")).not.toContain("select it first");
  });
  it("explains editing one layer alone", () => {
    expect(layerEditHint(null)).toBe("Pick the layer to edit in the Layers panel.");
    expect(layerEditHint("Anna")).toContain("edits only “Anna”");
    expect(layerEditHint("Anna", ["Tom"])).toContain("Tom keeps the look of their reference.");
  });
  it("composes references into a new image", () => {
    expect(promptPlaceholder("generate", false, ["Anna", "Tom"])).toBe("Where are Anna and Tom? e.g. riding a roller coaster together");
    expect(operationHint("generate", false, false, "object", ["Anna", "Tom"])).toBe("Describe a new scene with Anna and Tom; each keeps the look of their reference.");
    expect(operationHint("generate", true, false, "object", ["Anna"])).toContain("new layer");
  });
  it("asks where the reference goes before a selection exists", () => {
    expect(operationHint("replace", true, false, "rectangle", ["Anna"])).toBe("Select where Anna should appear. Leave room for the whole body and its shadow.");
  });

  it("explains that the selection sets the subject's size once selected", () => {
    expect(operationHint("replace", true, true, "rectangle", ["Anna"])).toContain("at the selection's size");
  });

  it("keeps the regular guidance for removal and without a reference", () => {
    expect(operationHint("remove", true, true, "object", ["Anna"])).toContain("shadow and glow");
    expect(operationHint("replace", true, false, "object")).toBe("Click an object to select it. Right-click to exclude a part.");
  });

  it("names the reference in the prompt placeholder", () => {
    expect(promptPlaceholder("replace", true, ["Anna"])).toBe("What is Anna doing here? e.g. sitting on the bench, reading");
    expect(promptPlaceholder("replace", true)).toBe("What would you like to see in your selection?");
  });

  it("names several references together", () => {
    expect(joinNames(["Anna", "Tom", "Mia"])).toBe("Anna, Tom and Mia");
    expect(promptPlaceholder("replace", true, ["Anna", "Tom"])).toBe("What are Anna and Tom doing here? e.g. sitting together on the bench");
    expect(operationHint("replace", true, false, "rectangle", ["Anna", "Tom"])).toBe("Select where Anna and Tom should appear. Leave room for everyone, with their shadows.");
    expect(operationHint("replace", true, true, "rectangle", ["Anna", "Tom"])).toMatch(/^Anna and Tom appear inside/);
  });
});
