import { describe, expect, it } from "vitest";
import { parseArtifacts } from "./api";

describe("parseArtifacts", () => {
  it("keeps server objects with size and timestamp", () => {
    expect(
      parseArtifacts([
        { path: "adapter/adapter_model.safetensors", bytes: 40_422_168, modified_at: "2026-11-11T04:50:16Z" },
      ]),
    ).toEqual([
      {
        path: "adapter/adapter_model.safetensors",
        bytes: 40_422_168,
        modified_at: "2026-11-11T04:50:16Z",
      },
    ]);
  });

  it("accepts plain path strings from older run records", () => {
    expect(parseArtifacts(["checkpoint-10", "adapter"])).toEqual([
      { path: "checkpoint-10", bytes: 0, modified_at: "" },
      { path: "adapter", bytes: 0, modified_at: "" },
    ]);
  });

  it("drops malformed entries instead of rendering blanks", () => {
    expect(parseArtifacts([null, 42, { bytes: 3 }, { path: "ok.bin", bytes: 3 }])).toEqual([
      { path: "ok.bin", bytes: 3, modified_at: "" },
    ]);
  });
});