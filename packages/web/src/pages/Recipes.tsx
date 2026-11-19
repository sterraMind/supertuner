import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { RECIPES, RECIPE_TIERS, type Recipe } from "@supertuner/core";
import { formatGib, formatMib } from "../lib/format";
import { useDocumentTitle, useResource } from "../lib/hooks";
import {
  ALL_TIERS,
  estimateDownload,
  familiesByKey,
  filterRecipes,
  groupRecipesByTier,
  licenceForFamily,
  type RecipeFilter,
} from "../lib/recipe-utils";
import type { ModelFamily } from "@supertuner/core";
import { getApiClient, formatApiError } from "../lib/api";
import { Badge, FeasibilityBadge } from "../components/StatusBadge";
import { EmptyState } from "../components/EmptyState";
import { SelectField, TextField } from "../components/Field";
import { Icon } from "../components/icons";

const MODALITIES: { value: RecipeFilter["kind"]; label: string }[] = [
  { value: "all", label: "All" },
  { value: "llm", label: "LLM" },
  { value: "video", label: "Video" },
];

export function Recipes() {
  useDocumentTitle("Recipes");
  const navigate = useNavigate();
  const client = getApiClient();
  const [filter, setFilter] = useState<RecipeFilter>({ tier: ALL_TIERS, kind: "all", search: "" });

  const list = useResource((signal) => client.recipes(signal), []);
  const plugins = useResource((signal) => client.plugins(signal), []);
  const families = useResource((signal) => client.families(signal), []);
  const system = useResource((signal) => client.system(signal), []);

  const apiRecipes = list.data;
  const usingFallback =
    list.status === "error" || (list.status === "ready" && (apiRecipes?.length ?? 0) === 0);
  const recipes: readonly Recipe[] = apiRecipes && apiRecipes.length > 0 ? apiRecipes : RECIPES;

  const familyMap = useMemo<Map<string, ModelFamily>>(() => {
    const all = families.data ? [...families.data.llm, ...families.data.video] : [];
    return familiesByKey(all);
  }, [families.data]);

  const pluginList = plugins.data?.plugins ?? [];
  const gpuVram = system.data?.env.gpus[0]?.vram_mib ?? null;
  const gpuName = system.data?.env.gpus[0]?.name ?? null;

  const filtered = filterRecipes(recipes, filter);
  const groups = groupRecipesByTier(filtered);

  return (
    <>
      <div className="page-head">
        <div className="page-head-copy">
          <span className="eyebrow">Curated recipes</span>
          <h1>Start from something known to fit</h1>
          <p>
            Each card is a checked combination of family, adapter, quantisation and resolution. Open one and the
            planner arrives pre-filled with the exact RunSpec — adjust anything before it compiles.
          </p>
        </div>
        <div className="page-actions">
          <Badge tone="neutral" icon="recipes">
            {recipes.length} recipes
          </Badge>
        </div>
      </div>

      {usingFallback ? (
        <div className="panel">
          <div className="panel-body">
            <p className="small muted">
              <Icon name="warning" size={14} /> Showing the bundled recipe list
              {list.status === "error" ? ` — /api/recipes failed (${formatApiError(list.error)})` : " — the server returned no recipes"}.
            </p>
          </div>
        </div>
      ) : null}

      <div className="filter-bar">
        <div className="segmented" role="group" aria-label="Modality">
          {MODALITIES.map((mode) => (
            <button
              key={mode.value}
              type="button"
              aria-pressed={filter.kind === mode.value}
              onClick={() => setFilter((previous) => ({ ...previous, kind: mode.value }))}
            >
              {mode.label}
            </button>
          ))}
        </div>
        <div className="filter-actions">
          <SelectField
            label="VRAM tier"
            value={filter.tier}
            options={[
              { value: ALL_TIERS, label: "All tiers" },
              ...RECIPE_TIERS.map((tier) => ({ value: tier, label: tier })),
            ]}
            onChange={(tier) => setFilter((previous) => ({ ...previous, tier }))}
          />
          <TextField
            label="Search"
            value={filter.search}
            placeholder="family, name, summary"
            onChange={(search) => setFilter((previous) => ({ ...previous, search }))}
          />
        </div>
      </div>

      {filtered.length === 0 ? (
        <EmptyState
          icon="recipes"
          title="No recipe matches those filters"
          description="Clear the search box or pick another tier. The planner still accepts a hand-built spec."
          action={
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => setFilter({ tier: ALL_TIERS, kind: "all", search: "" })}
            >
              Clear filters
            </button>
          }
        />
      ) : (
        <div className="stack">
          {groups.map((group) => (
            <section key={group.tier} className="recipe-group" aria-labelledby={`tier-${group.tier}`}>
              <div className="recipe-group-head">
                <h2 id={`tier-${group.tier}`}>{group.tier}</h2>
                <span className="small faint">
                  {group.llm.length + group.video.length} recipe{group.llm.length + group.video.length === 1 ? "" : "s"}
                </span>
              </div>
              {group.llm.length > 0 ? (
                <div className="stack-sm">
                  <span className="eyebrow">LLM</span>
                  <div className="recipe-grid">
                    {group.llm.map((recipe) => (
                      <RecipeCard
                        key={recipe.id}
                        recipe={recipe}
                        family={familyMap.get(recipe.family)}
                        licence={licenceForFamily(pluginList, recipe.family)}
                        gpuVram={gpuVram}
                        gpuName={gpuName}
                        onOpen={() => navigate(`/planner?recipe=${encodeURIComponent(recipe.id)}`)}
                      />
                    ))}
                  </div>
                </div>
              ) : null}
              {group.video.length > 0 ? (
                <div className="stack-sm">
                  <span className="eyebrow">Video</span>
                  <div className="recipe-grid">
                    {group.video.map((recipe) => (
                      <RecipeCard
                        key={recipe.id}
                        recipe={recipe}
                        family={familyMap.get(recipe.family)}
                        licence={licenceForFamily(pluginList, recipe.family)}
                        gpuVram={gpuVram}
                        gpuName={gpuName}
                        onOpen={() => navigate(`/planner?recipe=${encodeURIComponent(recipe.id)}`)}
                      />
                    ))}
                  </div>
                </div>
              ) : null}
            </section>
          ))}
        </div>
      )}
    </>
  );
}

interface RecipeCardProps {
  recipe: Recipe;
  family: ModelFamily | undefined;
  licence: ReturnType<typeof licenceForFamily>;
  gpuVram: number | null;
  gpuName: string | null;
  onOpen: () => void;
}

function RecipeCard({ recipe, family, licence, gpuVram, gpuName, onOpen }: RecipeCardProps) {
  const config = recipe.config;
  const download = estimateDownload(recipe, family);
  // Only a verified recipe has an end-to-end run behind it. Compile previews
  // and unverified entries may fit the training step yet fail on the cache step.
  const fits = gpuVram === null ? null : recipe.status === "verified" && recipe.vram_mib <= gpuVram;
  const isVideo = recipe.kind === "video";
  const resolution = isVideo
    ? `${config.width ?? "?"}×${config.height ?? "?"} · ${config.frames ?? "?"} frames`
    : `seq ${config.seq_len ?? "?"} tokens`;

  return (
    <article className="recipe-card" aria-labelledby={`recipe-${recipe.id}`}>
      <div className="spread">
        <div className="stack-sm">
          <h3 id={`recipe-${recipe.id}`} className="recipe-title">
            {recipe.name}
          </h3>
          <span className="small faint mono">{recipe.id}</span>
        </div>
        <Badge
          tone={
            recipe.status === "verified"
              ? "ok"
              : recipe.status === "compile_only"
                ? "warn"
                : "neutral"
          }
        >
          {recipe.status === "verified"
            ? "已验证"
            : recipe.status === "compile_only"
              ? "仅编译预览"
              : "未验证"}
        </Badge>
        {recipe.status === "verified" ? (
          <FeasibilityBadge
            fits={fits}
            detail={
              gpuVram === null
                ? "GPU not reported"
                : `plan needs ${formatMib(recipe.vram_mib)} of ${formatMib(gpuVram)}${gpuName ? ` on ${gpuName}` : ""}`
            }
          />
        ) : null}
      </div>

      <p className="recipe-summary">{recipe.summary}</p>

      <dl className="recipe-specs">
        <div className="recipe-spec">
          <dt>stage</dt>
          <dd>{config.stage}</dd>
        </div>
        <div className="recipe-spec">
          <dt>adapter</dt>
          <dd>{config.adapter ?? "lora"}</dd>
        </div>
        <div className="recipe-spec">
          <dt>quant</dt>
          <dd>{config.quant_base ?? "none"}</dd>
        </div>
        <div className="recipe-spec">
          <dt>{isVideo ? "resolution" : "context"}</dt>
          <dd className="mono">{resolution}</dd>
        </div>
        <div className="recipe-spec">
          <dt>rank</dt>
          <dd className="mono">
            {config.lora_rank ?? 16} / {config.lora_alpha ?? 16}
          </dd>
        </div>
        <div className="recipe-spec">
          <dt>RAM</dt>
          <dd className="mono">{formatGib(recipe.ram_mib, 0)}</dd>
        </div>
      </dl>

      <div className="recipe-foot">
        <span className="small">
          <Icon name="download" size={13} />{" "}
          {download ? `≈ ${formatGib(download.gib * 1024, 1)} download` : "download size unknown"}
        </span>
        <span className="small muted">{download?.basis ?? "no family metadata"}</span>
      </div>

      <p className="licence-note">
        <Icon name="warning" size={13} />{" "}
        {`${recipe.model_license}（模型权重许可）`}
        {licence ? ` · 工具 ${licence.plugin.name}（${licence.license}）` : ""}
      </p>

      {recipe.caveats.length > 0 ? (
        <ul className="stack-sm small muted" aria-label="Caveats">
          {recipe.caveats.slice(0, 3).map((caveat) => (
            <li key={caveat}>{caveat}</li>
          ))}
        </ul>
      ) : null}

      <button type="button" className="btn btn-primary btn-block" onClick={onOpen}>
        <Icon name="planner" size={14} />
        Open in planner
      </button>
    </article>
  );
}