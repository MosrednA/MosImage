import { describe, expect, it } from "vitest";
import { operationHint, promptPlaceholder } from "./operation";

describe("reference guidance", () => {
  it("asks where the reference goes before a selection exists", () => {
    expect(operationHint("replace", true, false, "rectangle", "Anna")).toBe("Select where Anna should appear. Leave room for the whole body and its shadow.");
  });

  it("explains that the selection sets the subject's size once selected", () => {
    expect(operationHint("replace", true, true, "rectangle", "Anna")).toContain("at the selection's size");
  });

  it("keeps the regular guidance for removal and without a reference", () => {
    expect(operationHint("remove", true, true, "object", "Anna")).toContain("shadow and glow");
    expect(operationHint("replace", true, false, "object")).toBe("Click an object to select it. Right-click to exclude a part.");
  });

  it("names the reference in the prompt placeholder", () => {
    expect(promptPlaceholder("replace", true, "Anna")).toBe("What is Anna doing here? e.g. sitting on the bench, reading");
    expect(promptPlaceholder("replace", true)).toBe("What would you like to see in your selection?");
  });
});
