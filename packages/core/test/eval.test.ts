import { describe, expect, it } from "vitest";
import { EvalReportEvent, EvalSpecSchema, RunSpecSchema, parseEventLine } from "../src/index.js";

describe("eval contract (v0.3 P0)", () => {
  it("defaults RunSpec.eval and parses EvalSpec", () => {
    const spec = RunSpecSchema.parse({
      schema_version: 1,
      config: { name: "t", family: "qwen3-0.6b", stage: "sft" },
    });
    expect(spec.eval.enabled).toBe(true);
    expect(spec.eval.holdout).toBe(true);
    expect(spec.eval.prompts_path).toBeNull();
    expect(EvalSpecSchema.parse({ metrics: ["clip_prompt_following"] }).metrics).toEqual(["clip_prompt_following"]);
  });

  it("parses eval_report event", () => {
    const event = EvalReportEvent.parse({
      ts: 1,
      type: "eval_report",
      status: "ok",
      path: "out/eval_report.json",
      report: { summary: { stability: { steps: 3 } } },
    });
    expect(event.type).toBe("eval_report");
    expect(event.report.summary).toBeDefined();
    const parsed = parseEventLine(JSON.stringify(event));
    expect(parsed.ok).toBe(true);
    if (parsed.ok) expect(parsed.event.type).toBe("eval_report");
  });
});
