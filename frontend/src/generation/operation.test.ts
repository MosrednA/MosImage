import { describe, expect, it } from "vitest";
import { joinNames, operationHint, promptPlaceholder } from "./operation";

describe("reference guidance", () => {
  it("explains generation layers without asking for a selection", () => {
    expect(promptPlaceholder("generate", true, ["Anna"])).toBe("Describe a new image to add as a layer…");
    expect(operationHint("generate", true, false, "object")).toContain("new layer");
    expect(operationHint("generate", false, false, "object")).toContain("Describe a new image");
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
