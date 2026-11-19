import { describe, expect, it } from "vitest";
import {
  AutoCandidateSchema,
  MemoryPlanSchema,
  type AutoCandidate,
  type MemoryPlan,
} from "../src/index.js";

/**
 * 统一自动选择（v0.2 T3 / G2 = AP-02）契约测试：确认 MemoryPlan 的
 * auto_candidates 字段（auto_budget 评估过的全部候选）在 TS 侧的
 * 形状与 engine 的 AutoCandidate 模型一致，且缺省为 []。
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

const candidate: AutoCandidate = {
  quant_base: "nf4",
  blocks_to_swap: 48,
  train_blocks: "20-49",
  feasible: true,
  headline_mib: 4156.5,
  note: "峰值 4157 MiB",
};

describe("统一自动选择（T3）契约", () => {
  it("AutoCandidateSchema 接受完整的候选记录", () => {
    const parsed = AutoCandidateSchema.parse(candidate);
    expect(parsed.quant_base).toBe("nf4");
    expect(parsed.blocks_to_swap).toBe(48);
    expect(parsed.train_blocks).toBe("20-49");
    expect(parsed.feasible).toBe(true);
    expect(parsed.headline_mib).toBeCloseTo(4156.5);
    expect(parsed.note).toBe("峰值 4157 MiB");
  });

  it("AutoCandidate 的 train_blocks 可为 null、note 可缺省", () => {
    const parsed = AutoCandidateSchema.parse({
      quant_base: "none",
      blocks_to_swap: 0,
      train_blocks: null,
      feasible: false,
      headline_mib: 9000,
    });
    expect(parsed.train_blocks).toBeNull();
    expect(parsed.note).toBe("");
  });

  it("MemoryPlanSchema 接受 auto_candidates 且字段名与 engine 一致", () => {
    const parsed = MemoryPlanSchema.parse({
      ...baseMemoryPlan(),
      auto_candidates: [
        candidate,
        { quant_base: "nf4", blocks_to_swap: 48, train_blocks: null, feasible: true, headline_mib: 4156.5, note: "峰值 4157 MiB" },
      ],
    }) as MemoryPlan;
    expect(parsed.auto_candidates).toHaveLength(2);
    expect(parsed.auto_candidates[0]).toEqual({
      quant_base: "nf4",
      blocks_to_swap: 48,
      train_blocks: "20-49",
      feasible: true,
      headline_mib: 4156.5,
      note: "峰值 4157 MiB",
    });
  });

  it("MemoryPlan 缺省 auto_candidates 为 []（legacy/LLM 计划不变）", () => {
    const parsed = MemoryPlanSchema.parse(baseMemoryPlan()) as MemoryPlan;
    expect(parsed.auto_candidates).toEqual([]);
  });

  it("auto_candidates 拒绝缺字段的行", () => {
    const bad = { ...baseMemoryPlan(), auto_candidates: [{ quant_base: "nf4" }] };
    expect(() => MemoryPlanSchema.parse(bad)).toThrow();
  });
});
