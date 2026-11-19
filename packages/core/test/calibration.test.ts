import { describe, expect, it } from "vitest";
import {
  MemoryPlanSchema,
  RunSpecSchema,
  calibrationKey,
  type MemoryPlan,
} from "../src/index.js";

/**
 * 跑后自校准（v0.2 T1）契约测试：确认 schema 接受 calibration 字段，
 * 且默认值与 calibrationKey 行为稳定。
 */

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

describe("calibration 契约", () => {
  it("RunSpecSchema 接受带 calibration 的 spec", () => {
    const spec = {
      config: { name: "t", family: "qwen3-0.6b", stage: "sft" },
      dataset: null,
      train: {},
      output: {},
      backend: null,
      calibration: { count: 3, avg_delta_pct: 2.5 },
    };
    const parsed = RunSpecSchema.parse(spec);
    expect(parsed.calibration).toEqual({ count: 3, avg_delta_pct: 2.5 });
  });

  it("RunSpecSchema 未提供 calibration 时默认 null", () => {
    const spec = {
      config: { name: "t", family: "qwen3-0.6b", stage: "sft" },
      dataset: null,
      train: {},
      output: {},
      backend: null,
    };
    const parsed = RunSpecSchema.parse(spec);
    expect(parsed.calibration).toBeNull();
  });

  it("RunSpecSchema 拒绝负数 count", () => {
    const spec = {
      config: { name: "t", family: "qwen3-0.6b", stage: "sft" },
      dataset: null,
      train: {},
      output: {},
      backend: null,
      calibration: { count: -1, avg_delta_pct: null },
    };
    expect(() => RunSpecSchema.parse(spec)).toThrow();
  });

  it("MemoryPlanSchema 接受 calibration 字段", () => {
    const plan = baseMemoryPlan();
    const parsed = MemoryPlanSchema.parse({ ...plan, calibration: { count: 5, avg_delta_pct: 1.5 } }) as MemoryPlan;
    expect(parsed.calibration).toEqual({ count: 5, avg_delta_pct: 1.5 });
    // 未提供时默认 null
    const bare = MemoryPlanSchema.parse(plan) as MemoryPlan;
    expect(bare.calibration).toBeNull();
  });

  it("calibrationKey 对同一五元组稳定，且顺序敏感", () => {
    const a = calibrationKey({ family: "f", stage: "sft", adapter: "lora", quant_base: "none", blocks_to_swap: 0 });
    const b = calibrationKey({ family: "f", stage: "sft", adapter: "lora", quant_base: "none", blocks_to_swap: 0 });
    expect(a).toBe(b);
    // 不同 quant_base 产生不同键
    const c = calibrationKey({ family: "f", stage: "sft", adapter: "lora", quant_base: "int8", blocks_to_swap: 0 });
    expect(c).not.toBe(a);
    // blocks_to_swap 以字符串形式进入键
    expect(a).toBe("f|sft|lora|none|0");
  });
});
