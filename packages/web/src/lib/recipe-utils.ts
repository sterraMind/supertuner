/**
 * Recipe presentation logic: grouping, filtering, download estimates, and the
 * licence lookup that has to come from plugin manifests because `Recipe` itself
 * carries no licence field.
 */

import { RECIPE_TIERS, type ModelFamily, type PluginManifest, type Recipe } from "@supertuner/core";

export interface RecipeFilter {
  tier: string;
  kind: "all" | "llm" | "video";
  search: string;
}

export const ALL_TIERS = "all";

export function filterRecipes(recipes: readonly Recipe[], filter: RecipeFilter): Recipe[] {
  const needle = filter.search.trim().toLowerCase();
  return recipes.filter((recipe) => {
    if (filter.tier !== ALL_TIERS && recipe.tier !== filter.tier) return false;
    if (filter.kind !== "all" && recipe.kind !== filter.kind) return false;
    if (needle.length === 0) return true;
    const haystack = `${recipe.name} ${recipe.family} ${recipe.summary} ${recipe.tier} ${recipe.kind}`.toLowerCase();
    return haystack.includes(needle);
  });
}

export interface RecipeGroup {
  tier: string;
  llm: Recipe[];
  video: Recipe[];
}

/** Tiers in ascending VRAM order, regardless of the order the API returns. */
export function groupRecipesByTier(recipes: readonly Recipe[]): RecipeGroup[] {
  const known = new Set<string>(RECIPE_TIERS);
  const tiers = [...RECIPE_TIERS, ...[...new Set(recipes.map((r) => r.tier))].filter((t) => !known.has(t))];
  return tiers
    .map((tier) => {
      const inTier = recipes.filter((recipe) => recipe.tier === tier);
      return {
        tier,
        llm: inTier.filter((recipe) => recipe.kind === "llm"),
        video: inTier.filter((recipe) => recipe.kind === "video"),
      };
    })
    .filter((group) => group.llm.length > 0 || group.video.length > 0);
}

/** Bytes per parameter for a base quantisation. */
export function bytesPerParam(quant: string | undefined): number {
  switch (quant) {
    case "nf4":
      return 0.55;
    case "nvfp4":
      return 0.5625;
    case "int8":
    case "fp8":
      return 1;
    default:
      return 2;
  }
}

export interface DownloadEstimate {
  gib: number;
  baseGib: number;
  extrasGib: number;
  basis: string;
}

/**
 * Weight download estimate from the family geometry: base transformer + text
 * encoder + VAE, each at its configured precision. Returns null when the family
 * metadata or parameter counts are unknown — better an em dash than a guess
 * presented as fact.
 */
export function estimateDownload(recipe: Recipe, family?: ModelFamily): DownloadEstimate | null {
  if (!family) return null;
  const params = family.pruned_params_b ?? family.params_b;
  if (params === null || params === undefined || params <= 0) return null;

  const baseBytes = params * 1e9 * bytesPerParam(recipe.config.quant_base);
  const textEncoderBytes =
    family.text_encoder_params_b > 0
      ? family.text_encoder_params_b * 1e9 * bytesPerParam(recipe.config.text_encoder_quant)
      : 0;
  const vaeBytes = family.vae_params_b > 0 ? family.vae_params_b * 1e9 * 2 : 0;
  const baseGib = baseBytes / 1024 ** 3;
  const extrasGib = (textEncoderBytes + vaeBytes) / 1024 ** 3;
  const rounded = (value: number): number => Math.round(value * 10) / 10;

  const basis =
    family.pruned_params_b !== null && family.pruned_params_b !== undefined
      ? `${family.pruned_params_b}B pruned weights`
      : `${params}B weights`;

  return { gib: rounded(baseGib + extrasGib), baseGib: rounded(baseGib), extrasGib: rounded(extrasGib), basis };
}

export interface FamilyLicence {
  license: string;
  plugin: PluginManifest;
}

/** First enabled plugin that claims this family, so the licence note has a source. */
export function licenceForFamily(
  plugins: readonly PluginManifest[],
  familyKey: string,
): FamilyLicence | null {
  const match = plugins.find((plugin) => plugin.families.includes(familyKey));
  if (!match) return null;
  return { license: match.license, plugin: match };
}

export function familiesByKey(families: readonly ModelFamily[]): Map<string, ModelFamily> {
  return new Map(families.map((family) => [family.key, family]));
}