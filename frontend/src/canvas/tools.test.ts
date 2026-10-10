import { describe, expect, it } from "vitest";
import { isSelectionTool, toolForKey } from "./tools";

describe("isSelectionTool", () => {
  it("covers the tools that select on the picture or mask a layer", () => {
    expect(["object", "brush", "eraser", "lasso", "rectangle"].every(tool => isSelectionTool(tool as never))).toBe(true);
  });

  it("leaves out Transform, Pan and Sketch, which do not select", () => {
    expect(isSelectionTool("transform")).toBe(false);
    expect(isSelectionTool("move")).toBe(false);
    expect(isSelectionTool("sketch")).toBe(false);
  });
});

describe("toolForKey", () => {
  it("follows Photoshop: W selects objects, V moves layers, H pans, regardless of case", () => {
    expect(toolForKey("w")).toBe("object");
    expect(toolForKey("W")).toBe("object");
    expect(toolForKey("v")).toBe("transform");
    expect(toolForKey("h")).toBe("move");
    expect(toolForKey("o")).toBeNull();
  });

  it("maps P to the Sketch pencil", () => {
    expect(toolForKey("p")).toBe("sketch");
  });
});
