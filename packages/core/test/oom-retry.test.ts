import { describe, expect, it } from "vitest";
import {
  RunConfigSchema,
  RunEventSchema,
  OOMRetryEvent,
} from "../src/index.js";

/**
 * OOM 有限次自动重试（v0.2 T4 / G4）契约测试：
 *  - `oom_retry` 事件能按 RunEventSchema 判别式联合正确解析，字段名与 engine
 *    侧 `events.py::Emitter.oom_retry` 完全一致（attempt / max_retries /
 *    reason / new_blocks_to_swap / new_quant_base / new_vram_budget_mib）。
 *  - RunConfig 的 max_oom_retries 默认值为 2。
 */

describe("OOM 有限次自动重试（T4）契约", () => {
  it("oom_retry 事件可经 RunEventSchema 解析且字段往返一致", () => {
    const raw = {
      ts: 1_700_000_000_000,
      type: "oom_retry",
      attempt: 1,
      max_retries: 2,
      reason: "step 'train' CUDA OOM",
      new_blocks_to_swap: 40,
      new_quant_base: "nf4",
      new_vram_budget_mib: 8192,
    };
    const parsed = RunEventSchema.parse(raw);
    expect(parsed.type).toBe("oom_retry");
    // 逐字段往返：对象本身不变。
    expect(parsed).toEqual(raw);
  });

  it("OOMRetryEvent 只接受完整的 oom_retry 负载", () => {
    const parsed = OOMRetryEvent.parse({
      ts: 1,
      type: "oom_retry",
      attempt: 2,
      max_retries: 2,
      reason: "CUDA OOM",
      new_blocks_to_swap: 48,
      new_quant_base: "nf4",
      new_vram_budget_mib: 7168,
    });
    expect(parsed.attempt).toBe(2);
    expect(parsed.new_blocks_to_swap).toBe(48);
    expect(parsed.new_quant_base).toBe("nf4");
    // 缺字段应被拒绝。
    expect(() =>
        OOMRetryEvent.parse({
          ts: 1,
          type: "oom_retry",
          attempt: 1,
          max_retries: 2,
          reason: "x",
          // new_blocks_to_swap / new_quant_base / new_vram_budget_mib 缺失
        }),
    ).toThrow();
  });

  it("RunConfigSchema 的 max_oom_retries 默认值为 2", () => {
    const parsed = RunConfigSchema.parse({
      name: "t4",
      family: "wan2.1-t2v-14b",
      stage: "sft",
    });
    expect(parsed.max_oom_retries).toBe(2);
  });

  it("RunConfigSchema 接受显式的 max_oom_retries（含 0 = 禁用）", () => {
    const parsed = RunConfigSchema.parse({
      name: "t4",
      family: "wan2.1-t2v-14b",
      stage: "sft",
      max_oom_retries: 0,
    });
    expect(parsed.max_oom_retries).toBe(0);
  });
});
