/** Artifact discovery under a run's `output/` directory. */

import { readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";

const MAX_DEPTH = 8;
const MAX_FILES = 5_000;

export const ArtifactSchema = z.object({
  /** POSIX-style path relative to `output/`. */
  path: z.string(),
  bytes: z.number().int().nonnegative(),
  modified_at: z.string(),
});

export type Artifact = z.infer<typeof ArtifactSchema>;

function walk(dir: string, prefix: string, depth: number, out: Artifact[]): void {
  if (depth > MAX_DEPTH || out.length >= MAX_FILES) return;
  let entries;
  try {
    entries = readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const entry of entries) {
    if (out.length >= MAX_FILES) return;
    const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(full, rel, depth + 1, out);
      continue;
    }
    if (!entry.isFile()) continue;
    try {
      const info = statSync(full);
      out.push({ path: rel, bytes: info.size, modified_at: info.mtime.toISOString() });
    } catch {
      // A file that vanished between readdir and stat is not an error.
    }
  }
}

/** Every regular file under `dir`, relative paths, sorted for stable responses. */
export function listArtifacts(dir: string): Artifact[] {
  const out: Artifact[] = [];
  walk(dir, "", 0, out);
  out.sort((a, b) => a.path.localeCompare(b.path));
  return out;
}

/** Just the relative paths; what the run record stores. */
export function listArtifactPaths(dir: string): string[] {
  return listArtifacts(dir).map((artifact) => artifact.path);
}