import { describe, expect, it } from "vitest";
import { RECIPES, RunSpecSchema } from "@supertuner/core";
import {
  defaultConfig,
  defaultDataset,
  defaultRunSpec,
  initialPlannerState,
  plannerReducer,
  specFromRecipe,
  validateSpec,
} from "./planner-state";

const recipe = RECIPES[0]!;

function validSpec() {
  return {
    ...defaultRunSpec(),
    config: { ...defaultConfig(), name: "smoke", family: "qwen-test" },
  };
}

describe("default spec", () => {
  it("parses against the core contract once a recipe fills it", () => {
    expect(RunSpecSchema.safeParse(specFromRecipe(recipe)).success).toBe(true);
  });

  it("is locally incomplete until a name and family are set", () => {
    const result = validateSpec(defaultRunSpec());
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.issues.length).toBeGreaterThan(0);
      expect(result.issues.join(" ")).toMatch(/name|family/);
    }
  });

  it("accepts a named spec", () => {
    const result = validateSpec(validSpec());
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.spec.config.name).toBe("smoke");
    }
  });

  it("reports out-of-range numbers with the field path", () => {
    const spec = validSpec();
    spec.config.lora_rank = 0;
    const result = validateSpec(spec);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.issues.join(" ")).toContain("lora_rank");
    }
  });
});

describe("specFromRecipe", () => {
  it("applies config and train values from the curated recipe", () => {
    const spec = specFromRecipe(recipe);
    expect(spec.config.family).toBe(recipe.family);
    expect(spec.config.stage).toBe(recipe.config.stage);
    expect(spec.config.name.length).toBeGreaterThan(0);
    for (const [key, value] of Object.entries(recipe.train)) {
      expect(spec.train[key as keyof typeof spec.train]).toEqual(value);
    }
    expect(validateSpec(spec).ok).toBe(true);
  });

  it("keeps defaults for fields the recipe omits", () => {
    const config = { ...recipe.config };
    delete config.adapter;
    const spec = specFromRecipe({ ...recipe, config });
    expect(spec.config.adapter).toBe(defaultConfig().adapter);
    expect(spec.dataset).toBeNull();
  });

  it("leaves the base dataset untouched", () => {
    const base = { ...defaultRunSpec(), dataset: defaultDataset("chat", "/data/chat.jsonl") };
    const spec = specFromRecipe(recipe, base);
    expect(spec.dataset?.path).toBe("/data/chat.jsonl");
  });
});

describe("plannerReducer", () => {
  it("prefills and resets the dirty flag", () => {
    const next = plannerReducer(initialPlannerState(), {
      type: "prefill",
      spec: specFromRecipe(recipe),
      recipeId: recipe.id,
    });
    expect(next.recipeId).toBe(recipe.id);
    expect(next.dirty).toBe(false);
    expect(next.spec.config.family).toBe(recipe.family);
  });

  it("patches config and marks the state dirty", () => {
    const next = plannerReducer(initialPlannerState(), {
      type: "patchConfig",
      patch: { batch: 4, grad_accum: 2 },
    });
    expect(next.spec.config.batch).toBe(4);
    expect(next.spec.config.grad_accum).toBe(2);
    expect(next.dirty).toBe(true);
  });

  it("auto-names the run when the family changes and the name is blank", () => {
    const next = plannerReducer(initialPlannerState(), {
      type: "patchConfig",
      patch: { family: "qwen3-8b" },
    });
    expect(next.spec.config.name).toBe("qwen3-8b-sft");
  });

  it("never overwrites an operator-supplied name", () => {
    const state = plannerReducer(initialPlannerState(), {
      type: "patchConfig",
      patch: { name: "my-run" },
    });
    const next = plannerReducer(state, { type: "patchConfig", patch: { family: "other" } });
    expect(next.spec.config.name).toBe("my-run");
  });

  it("creates a dataset when patching one that does not exist yet", () => {
    const next = plannerReducer(initialPlannerState(), {
      type: "patchDataset",
      patch: { path: "/data/corpus.jsonl", kind: "chat" },
    });
    expect(next.spec.dataset?.path).toBe("/data/corpus.jsonl");
    expect(next.spec.dataset?.kind).toBe("chat");
    expect(next.spec.dataset?.format).toBe("auto");
  });

  it("detaches and re-attaches a dataset", () => {
    const attached = plannerReducer(initialPlannerState(), {
      type: "setDataset",
      dataset: defaultDataset("text", "/data/t.txt"),
    });
    expect(attached.spec.dataset).not.toBeNull();
    const detached = plannerReducer(attached, { type: "setDataset", dataset: null });
    expect(detached.spec.dataset).toBeNull();
  });

  it("patches train and output independently", () => {
    const next = plannerReducer(
      plannerReducer(initialPlannerState(), { type: "patchTrain", patch: { max_steps: 42 } }),
      { type: "patchOutput", patch: { export: ["gguf"], dir: "/out" } },
    );
    expect(next.spec.train.max_steps).toBe(42);
    expect(next.spec.output.export).toEqual(["gguf"]);
    expect(next.spec.output.dir).toBe("/out");
  });

  it("sets and clears the backend override", () => {
    const set = plannerReducer(initialPlannerState(), { type: "setBackend", backend: "diffusers" });
    expect(set.spec.backend).toBe("diffusers");
    expect(plannerReducer(set, { type: "setBackend", backend: null }).spec.backend).toBeNull();
  });

  it("resets to a fresh default state", () => {
    const dirty = plannerReducer(initialPlannerState(), { type: "patchConfig", patch: { name: "x" } });
    const reset = plannerReducer(dirty, { type: "reset" });
    expect(reset.recipeId).toBeNull();
    expect(reset.dirty).toBe(false);
    expect(reset.spec).toEqual(defaultRunSpec());
  });
});