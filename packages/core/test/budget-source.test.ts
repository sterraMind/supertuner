import { describe, expect, it } from "vitest";
import {
  MemoryPlanSchema,
  RunConfigSchema,
  type MemoryPlan,
  type RunConfig,
} from "../src/index.js";

/**
 * 实测进预算（v0.2 T2 / G3 = AP-01）契约测试：确认 RunConfig 的
 * budget_source 输入字段与 MemoryPlan 的解析结果字段两侧形状一致。
 */

function baseRunConfig(): Record<string, unknown> {
  return { name: "t", family: "qwen3-0.6b", stage: "sft" };
}

function baseMemoryPlan(): Record<string, unknown> {
  return {
    feasible: true,
    vram_budget_mib: 8192,
    training_peak_mib: 4000,
    sampling_peak_mib: 3000,
    headline_mib: 4000,
    headroom_mib: 4192,
    lines: [],
    risks: [],
    tokens_per_step: 100,
    notes: [],
  };
}

describe("budget source 契约", () => {
  it("RunConfigSchema 默认 budget_source 为 auto", () => {
    const parsed = RunConfigSchema.parse(baseRunConfig()) as RunConfig;
    expect(parsed.budget_source).toBe("auto");
  });

  it("RunConfigSchema 接受 manual", () => {
    const parsed = RunConfigSchema.parse({
      ...baseRunConfig(),
      budget_source: "manual",
    }) as RunConfig;
    expect(parsed.budget_source).toBe("manual");
  });

  it("RunConfigSchema 拒绝非法 budget_source", () => {
    expect(() =>
      RunConfigSchema.parse({ ...baseRunConfig(), budget_source: "measured" })
    ).toThrow();
    expect(() =>
      RunConfigSchema.parse({ ...baseRunConfig(), budget_source: "bogus" })
    ).toThrow();
  });

  it("MemoryPlanSchema 接受 measured 来源与实测值", () => {
    const plan = baseMemoryPlan();
    const parsed = MemoryPlanSchema.parse({
      ...plan,
      budget_source: "measured",
      budget_measured_vram_mib: 24576,
      budget_measured_ram_mib: 65536,
    }) as MemoryPlan;
    expect(parsed.budget_source).toBe("measured");
    expect(parsed.budget_measured_vram_mib).toBe(24576);
    expect(parsed.budget_measured_ram_mib).toBe(65536);
  });

  it("MemoryPlanSchema 默认 manual / null 实测值", () => {
    const parsed = MemoryPlanSchema.parse(baseMemoryPlan()) as MemoryPlan;
    expect(parsed.budget_source).toBe("manual");
    expect(parsed.budget_measured_vram_mib).toBeNull();
    expect(parsed.budget_measured_ram_mib).toBeNull();
  });

  it("MemoryPlanSchema 拒绝非法 budget_source", () => {
    expect(() =>
      MemoryPlanSchema.parse({ ...baseMemoryPlan(), budget_source: "auto" })
    ).toThrow();
  });
});
