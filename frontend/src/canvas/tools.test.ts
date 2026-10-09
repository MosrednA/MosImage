import { describe, expect, it } from "vitest";
import { isSelectionTool, toolForKey } from "./tools";

describe("isSelectionTool", () => {
  it("covers the tools that select on the picture or mask a layer", () => {
    expect(["object", "brush", "eraser", "lasso", "rectangle"].every(tool => isSelectionTool(tool as never))).toBe(true);
  });

  it("leaves out Transform and Pan, which keep the layer selected", () => {
    expect(isSelectionTool("transform")).toBe(false);
    expect(isSelectionTool("move")).toBe(false);
  });
});

describe("toolForKey", () => {
  it("maps O to the Object tool regardless of case", () => {
    expect(toolForKey("o")).toBe("object");
    expect(toolForKey("O")).toBe("object");
  });
});
