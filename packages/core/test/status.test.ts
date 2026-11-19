import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { RECIPE_TIERS, RECIPES } from "../src/recipes.js";

/**
 * T5 (AP-07) 对外声明对齐 / external-declaration alignment:
 * the "Truth about the hardware" table in README.md and the recipe list in
 * src/recipes.ts are two external surfaces that must not drift apart.
 *
 * This test parses the ACTUAL README truth table (no golden copy of the
 * markdown) and asserts, against RECIPES:
 *
 *  1. every recipe's `status` is one of verified / compile_only / unverified;
 *  2. the (tier, family) row set of the truth table is 1:1 with the
 *     (tier, family) row set of RECIPES — the only extra row allowed is the
 *     "not supported" wan2.2-ti2v-5b row, which must have no recipe;
 *  3. wan2.2-ti2v-5b (Wan2.2 TI2V 5B) is not advertised as a trainable recipe.
 */

const STATUSES = new Set<string>(["verified", "compile_only", "unverified"]);
const FIVE_B = "wan2.2-ti2v-5b";

/**
 * Display names by which each family appears in the README "Configuration"
 * column. A row is attributed to exactly one family; a row matching none or
 * more than one name is a drift signal and fails the test.
 */
const FAMILY_DISPLAY_NAMES: Record<string, readonly string[]> = {
  "qwen3-0.6b": ["Qwen3-0.6B"],
  "qwen3-1.7b": ["Qwen3-1.7B"],
  "qwen3-4b": ["Qwen3-4B"],
  "qwen3-8b": ["Qwen3-8B"],
  "wan2.1-t2v-1.3b": ["Wan2.1 1.3B"],
  "wan2.1-t2v-14b": ["Wan2.1 14B"],
  "minimax-h3": ["MiniMax-H3"],
  "wan2.2-ti2v-5b": ["Wan2.2"],
};

interface TruthRow {
  tier: string;
  configuration: string;
  status: string;
}

function readRootReadme(): string {
  const readme = new URL("../../../README.md", import.meta.url);
  return readFileSync(fileURLToPath(readme), "utf-8");
}

function truthTableSection(readme: string): string {
  const marker = "## Truth about the hardware";
  const start = readme.indexOf(marker);
  if (start === -1) {
    throw new Error("README.md: '## Truth about the hardware' section not found");
  }
  const rest = readme.slice(start);
  const end = rest.indexOf("\n## ", 1);
  return end === -1 ? rest : rest.slice(0, end);
}

function parseTruthRows(section: string): TruthRow[] {
  const rows: TruthRow[] = [];
  for (const rawLine of section.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line.startsWith("|")) continue;
    const cells = line.split("|").map((cell) => cell.trim());
    // "| Card | Configuration | Status |" splits to ["", card, config, status, ""]
    if (cells.length < 4) continue;
    const [tier, configuration, status] = [cells[1], cells[2], cells[3]];
    if (!tier || configuration === undefined || status === undefined) continue;
    if (tier === "Card" || tier === "---") continue;
    rows.push({ tier, configuration, status });
  }
  return rows;
}

function familyOf(configuration: string): string | undefined {
  const matches = Object.entries(FAMILY_DISPLAY_NAMES)
    .filter(([, names]) => names.some((name) => configuration.includes(name)))
    .map(([family]) => family);
  return matches.length === 1 ? matches[0] : undefined;
}

function countBy(items: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const item of items) {
    counts.set(item, (counts.get(item) ?? 0) + 1);
  }
  return counts;
}

function sortedEntries(map: Map<string, number>): Array<[string, number]> {
  return [...map.entries()].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
}

function rowKey(tier: string, family: string): string {
  return `${tier} | ${family}`;
}

const README_ROWS = parseTruthRows(truthTableSection(readRootReadme()));

describe("external declaration alignment (T5 / AP-07)", () => {
  it("every recipe status is verified, compile_only, or unverified", () => {
    expect(RECIPES.length).toBeGreaterThan(0);
    const bad = RECIPES.filter((recipe) => !STATUSES.has(recipe.status));
    expect(bad.map((recipe) => `${recipe.id}: ${recipe.status}`)).toEqual([]);
  });

  it("recipe (tier, family) rows match the README truth table 1:1", () => {
    // Every recipe family must be nameable in the README, and every table row
    // must attribute to exactly one family.
    for (const recipe of RECIPES) {
      expect(
        Object.keys(FAMILY_DISPLAY_NAMES),
        `recipe ${recipe.id} has no README display name for family ${recipe.family}`,
      ).toContain(recipe.family);
    }
    const attributed = README_ROWS.map((row) => ({
      row,
      family: familyOf(row.configuration),
    }));
    for (const { row, family } of attributed) {
      expect(
        family,
        `truth-table row cannot be attributed to exactly one family: ${row.configuration}`,
      ).toBeDefined();
    }

    const tablePairs = attributed.map(({ family, row }) => rowKey(row.tier, family!));
    const recipePairs = RECIPES.map((recipe) => rowKey(recipe.tier, recipe.family));

    const tableCount = countBy(tablePairs);
    const expected = countBy(recipePairs);
    const fiveB = rowKey("16GB", FIVE_B);
    // The table may carry exactly one extra row: the not-supported 5B row,
    // which has no recipe behind it.
    expected.set(fiveB, (expected.get(fiveB) ?? 0) + 1);

    const actual = sortedEntries(tableCount);
    const want = sortedEntries(expected);
    expect(actual, `README truth table rows drifted from recipes.ts:\n table: ${JSON.stringify(actual)}\n want:  ${JSON.stringify(want)}`).toEqual(want);
  });

  it("wan2.2-ti2v-5b is not advertised as a trainable recipe", () => {
    expect(RECIPES.filter((recipe) => recipe.family === FIVE_B)).toEqual([]);
  });

  it("the 5B row in the truth table is the not-supported row", () => {
    const fiveBRows = README_ROWS.filter((row) => familyOf(row.configuration) === FIVE_B);
    expect(fiveBRows).toHaveLength(1);
    expect(fiveBRows[0].tier).toBe("16GB");
    expect(fiveBRows[0].status.toLowerCase()).toContain("not supported");
  });

  it("every truth-table tier is a real recipe tier", () => {
    for (const row of README_ROWS) {
      expect(RECIPE_TIERS as readonly string[]).toContain(row.tier);
    }
  });
});
