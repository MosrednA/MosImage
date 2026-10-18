import { describe, expect, it } from "vitest";
import { ENGINE_UNREACHABLE, errorMessage } from "./errors";

describe("error messages", () => {
  it("replaces fetch's own words when nothing answered", () => {
    for (const raw of ["Failed to fetch", "Load failed", "NetworkError when attempting to fetch resource."]) {
      expect(errorMessage(new TypeError(raw))).toBe(ENGINE_UNREACHABLE);
    }
  });

  it("keeps other errors as they are", () => {
    expect(errorMessage(new TypeError("Cannot read properties of undefined (reading 'fetch')"))).toBe("Cannot read properties of undefined (reading 'fetch')");
    expect(errorMessage(new Error("Choose a PNG, JPG or WebP image."))).toBe("Choose a PNG, JPG or WebP image.");
    expect(errorMessage("nope")).toBe("The operation failed.");
  });
});
