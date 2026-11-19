import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { calibrationKey } from "@supertuner/core";
import { RunStore, type CalibrationInput } from "../src/db.js";

/** 跑后自校准（v0.2 T1）：calibrations 表的聚合行为。 */

function makeInput(over: Partial<CalibrationInput> = {}): CalibrationInput {
  return {
    family: "qwen3-0.6b",
    stage: "sft",
    adapter: "lora",
    quant_base: "none",
    blocks_to_swap: 0,
    delta_pct: null,
    ...over,
  };
}

let store: RunStore;

beforeEach(async () => {
  const dir = await mkdtemp(join(tmpdir(), "calib-test-"));
  store = new RunStore(join(dir, "calib.db"));
});

afterEach(() => {
  store.close();
});

describe("upsertCalibration", () => {
  it("同一校准键重复运行：count 累加、sum 与 avg 正确", () => {
    store.upsertCalibration(makeInput({ delta_pct: 2.0 }));
    store.upsertCalibration(makeInput({ delta_pct: 4.0 }));

    const row = store.listCalibrations()[0]!;
    expect(row.count).toBe(2);
    expect(row.sum_delta_pct).toBe(6);
    expect(row.avg_delta_pct).toBeCloseTo(3.0);
    // key 五元组可复现
    const key = calibrationKey({
      family: "qwen3-0.6b",
      stage: "sft",
      adapter: "lora",
      quant_base: "none",
      blocks_to_swap: 0,
    });
    expect(row.key).toBe(key);
    expect(store.getCalibration(key)?.count).toBe(2);
  });

  it("不同校准键彼此独立，不串档", () => {
    store.upsertCalibration(makeInput({ delta_pct: 2.0 }));
    store.upsertCalibration(makeInput({ delta_pct: 10.0, quant_base: "int8" }));
    store.upsertCalibration(makeInput({ delta_pct: 8.0, blocks_to_swap: 3 }));

    const rows = store.listCalibrations();
    expect(rows).toHaveLength(3);
    const int8 = rows.find((r) => r.quant_base === "int8")!;
    const swapped = rows.find((r) => r.blocks_to_swap === 3)!;
    expect(int8.count).toBe(1);
    expect(int8.avg_delta_pct).toBe(10);
    expect(swapped.count).toBe(1);
    expect(swapped.avg_delta_pct).toBe(8);
    // 基准键只被写了一次
    const base = rows.find((r) => r.quant_base === "none" && r.blocks_to_swap === 0)!;
    expect(base.count).toBe(1);
    expect(base.avg_delta_pct).toBe(2);
  });

  it("delta_pct 为 null 时占计数但不贡献求和", () => {
    store.upsertCalibration(makeInput({ delta_pct: null }));
    store.upsertCalibration(makeInput({ delta_pct: 4.0 }));

    const row = store.listCalibrations()[0]!;
    expect(row.count).toBe(2);
    expect(row.sum_delta_pct).toBe(4);
    expect(row.avg_delta_pct).toBeCloseTo(2.0);
  });

  it("getCalibration 对未知键返回 null，命中时返回聚合行", () => {
    expect(store.getCalibration("no-such-key")).toBeNull();
    store.upsertCalibration(makeInput({ delta_pct: 1.5 }));
    const key = calibrationKey({
      family: "qwen3-0.6b",
      stage: "sft",
      adapter: "lora",
      quant_base: "none",
      blocks_to_swap: 0,
    });
    const row = store.getCalibration(key);
    expect(row).not.toBeNull();
    expect(row?.count).toBe(1);
    expect(row?.avg_delta_pct).toBe(1.5);
  });
});
