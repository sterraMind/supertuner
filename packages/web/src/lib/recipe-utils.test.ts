import { describe, expect, it } from "vitest";
import {
  RECIPES,
  RECIPE_TIERS,
  type ModelFamily,
  type PluginManifest,
  type Recipe,
} from "@supertuner/core";
import {
  ALL_TIERS,
  bytesPerParam,
  estimateDownload,
  familiesByKey,
  filterRecipes,
  groupRecipesByTier,
  licenceForFamily,
} from "./recipe-utils";

const emptyFilter = { tier: ALL_TIERS, kind: "all", search: "" } as const;

describe("filterRecipes", () => {
  it("returns every recipe for the empty filter", () => {
    expect(filterRecipes(RECIPES, emptyFilter)).toHaveLength(RECIPES.length);
  });

  it("filters by modality", () => {
    const video = filterRecipes(RECIPES, { ...emptyFilter, kind: "video" });
    expect(video.length).toBeGreaterThan(0);
    expect(video.every((recipe) => recipe.kind === "video")).toBe(true);

    const llm = filterRecipes(RECIPES, { ...emptyFilter, kind: "llm" });
    expect(llm.length).toBeGreaterThan(0);
    expect(llm.every((recipe) => recipe.kind === "llm")).toBe(true);
    expect(video.length + llm.length).toBe(RECIPES.length);
  });

  it("filters by VRAM tier", () => {
    const tier = RECIPES[0]?.tier ?? "8GB";
    const filtered = filterRecipes(RECIPES, { ...emptyFilter, tier });
    expect(filtered.length).toBeGreaterThan(0);
    expect(filtered.every((recipe) => recipe.tier === tier)).toBe(true);
  });

  it("searches name, family and summary case-insensitively", () => {
    const recipe = RECIPES[0]!;
    const byName = filterRecipes(RECIPES, { ...emptyFilter, search: recipe.name.slice(0, 6).toUpperCase() });
    expect(byName.map((entry) => entry.id)).toContain(recipe.id);

    const byFamily = filterRecipes(RECIPES, { ...emptyFilter, search: recipe.family });
    expect(byFamily.every((entry) => entry.family === recipe.family || entry.kind === recipe.kind)).toBe(true);
    expect(byFamily.map((entry) => entry.id)).toContain(recipe.id);
  });

  it("returns nothing for an impossible query", () => {
    expect(filterRecipes(RECIPES, { ...emptyFilter, search: "zzz-no-such-recipe-zzz" })).toHaveLength(0);
  });
});

describe("groupRecipesByTier", () => {
  it("orders groups by the curated tier order and splits modality", () => {
    const groups = groupRecipesByTier(RECIPES);
    const known = groups.filter((group) => (RECIPE_TIERS as readonly string[]).includes(group.tier));
    const orderedTiers = known.map((group) => group.tier);
    const expectedOrder = RECIPE_TIERS.filter((tier) => orderedTiers.includes(tier));
    expect(orderedTiers).toEqual(expectedOrder);
    for (const group of groups) {
      if (group.llm.length === 0 && group.video.length === 0) {
        throw new Error(`group ${group.tier} should not be empty`);
      }
    }
  });

  it("appends tiers the curated list does not know about", () => {
    const custom: Recipe = { ...RECIPES[0]!, id: "custom-64gb", tier: "64GB" };
    const groups = groupRecipesByTier([...RECIPES, custom]);
    expect(groups[groups.length - 1]?.tier).toBe("64GB");
    expect(groups[groups.length - 1]?.llm[0]?.id).toBe("custom-64gb");
  });
});

const family: ModelFamily = {
  key: "qwen-test",
  kind: "llm",
  params_b: 8,
  hidden: 4096,
  layers: 36,
  heads: 32,
  kv_heads: 8,
  head_dim: 128,
  ffn: 14336,
  vocab: 151936,
  tie_embeddings: true,
  text_encoder_params_b: 0,
  vae_params_b: 0,
  patch: [1, 2, 2],
  vae_temporal_stride: 4,
  vae_spatial_stride: 8,
  latent_channels: 16,
  fps: 0,
  pruned_params_b: null,
  adaln_params_b: 0,
  supports_block_swap: false,
  text_encoder_layers: 0,
  approx: false,
};

const plugin: PluginManifest = {
  id: "qwen-plugin",
  name: "Qwen families",
  version: "0.1.0",
  kind: "model",
  description: "",
  capabilities: [],
  families: ["qwen-test"],
  entry_point: "qwen_plugin:register",
  requires: [],
  license: "Apache-2.0",
  homepage: "",
  enabled: true,
  experimental: false,
  priority: 100,
};

describe("download estimate", () => {
  it("prices quantised weights below fp16", () => {
    const recipe = RECIPES[0]!;
    const nf4 = estimateDownload({ ...recipe, config: { ...recipe.config, quant_base: "nf4" } }, family);
    const full = estimateDownload({ ...recipe, config: { ...recipe.config, quant_base: "none" } }, family);
    expect(nf4).not.toBeNull();
    expect(full).not.toBeNull();
    expect(nf4!.gib).toBeLessThan(full!.gib);
    expect(nf4!.basis.length).toBeGreaterThan(0);
  });

  it("uses a sane bytes-per-param ratio per quantisation", () => {
    expect(bytesPerParam("nf4")).toBeLessThan(bytesPerParam("none"));
    expect(bytesPerParam("fp8")).toBe(1);
    expect(bytesPerParam("int8")).toBe(1);
  });

  it("prefers pruned parameters when the family reports them", () => {
    const recipe = RECIPES[0]!;
    const pruned = estimateDownload(recipe, { ...family, pruned_params_b: 2 });
    const unpruned = estimateDownload(recipe, family);
    expect(pruned!.gib).toBeLessThan(unpruned!.gib);
  });
});

describe("licence and family helpers", () => {
  it("finds the licence via a plugin that claims the family", () => {
    expect(licenceForFamily([plugin], "qwen-test")).toMatchObject({ license: "Apache-2.0" });
    expect(licenceForFamily([plugin], "unknown-family")).toBeNull();
    expect(licenceForFamily([], "qwen-test")).toBeNull();
  });

  it("indexes families by key", () => {
    const map = familiesByKey([family]);
    expect(map.get("qwen-test")).toBe(family);
    expect(map.size).toBe(1);
  });
});