/**
 * Plugin contract shared by the engine and the UI.
 *
 * The engine discovers plugins through Python entry points (group
 * `supertuner.plugins`); it then reports their manifests as JSON. The UI never
 * executes plugin code — it gates controls on declared capabilities. This keeps
 * one registration mechanism (Python packaging) instead of two divergent ones.
 */

import { z } from "zod";

export const PLUGIN_KINDS = ["model", "dataset", "strategy", "optimizer", "exporter", "ui", "backend"] as const;

export const CapabilitySchema = z.enum([
  "train.llm.pretrain",
  "train.llm.sft",
  "train.llm.preference",
  "train.llm.rl",
  "train.llm.lora",
  "train.llm.full",
  "train.video.pretrain",
  "train.video.lora",
  "train.video.full",
  "train.distill.dmd",
  "train.distill.dmad",
  "train.video.distill.dmd",
  "train.video.distill.dmad",
  "data.text",
  "data.chat",
  "data.preference",
  "data.video",
  "data.image",
  "model.llm",
  "model.video",
  "export.safetensors",
  "export.diffusers",
  "export.gguf",
  "export.onnx",
]);

export type Capability = z.infer<typeof CapabilitySchema>;

export const PluginManifestSchema = z.object({
  id: z.string().regex(/^[a-z0-9][a-z0-9._-]*$/),
  name: z.string().min(1),
  version: z.string().min(1),
  kind: z.enum(PLUGIN_KINDS),
  description: z.string().default(""),
  capabilities: z.array(CapabilitySchema).default([]),
  families: z.array(z.string()).default([]),
  entry_point: z.string().default(""),
  requires: z.array(z.string()).default([]),
  license: z.string().default("unknown"),
  homepage: z.string().default(""),
  enabled: z.boolean().default(true),
  experimental: z.boolean().default(false),
  priority: z.number().int().default(100),
});

export type PluginManifest = z.infer<typeof PluginManifestSchema>;

export const PluginsResponseSchema = z.object({
  plugins: z.array(PluginManifestSchema),
  errors: z.array(z.object({ entry_point: z.string(), reason: z.string() })).default([]),
});

export type PluginsResponse = z.infer<typeof PluginsResponseSchema>;

export interface PluginRegistryEntry extends PluginManifest {
  /** resolved at load time, never sent to the browser */
  source: string;
}

export class PluginRegistry {
  private readonly byId = new Map<string, PluginRegistryEntry>();

  register(manifest: PluginManifest, source: string): void {
    const existing = this.byId.get(manifest.id);
    if (existing && existing.version !== manifest.version) {
      throw new Error(
        `plugin id conflict: "${manifest.id}" ${existing.version} (${existing.source}) vs ${manifest.version} (${source})`,
      );
    }
    this.byId.set(manifest.id, { ...manifest, source });
  }

  list(): PluginRegistryEntry[] {
    return [...this.byId.values()].sort((a, b) => a.id.localeCompare(b.id));
  }

  enabled(): PluginRegistryEntry[] {
    return this.list().filter((p) => p.enabled);
  }

  get(id: string): PluginRegistryEntry | undefined {
    return this.byId.get(id);
  }

  supports(capability: Capability): boolean {
    return this.enabled().some((p) => p.capabilities.includes(capability));
  }

  familiesFor(capability: Capability): string[] {
    const out = new Set<string>();
    for (const plugin of this.enabled()) {
      if (plugin.capabilities.includes(capability)) plugin.families.forEach((f) => out.add(f));
    }
    return [...out].sort();
  }
}

/**
 * Capabilities a stage/adapter combination needs, so the UI can disable an
 * impossible combination before a run is even planned.
 */
export function requiredCapabilities(stage: string, kind: "llm" | "video", adapter: string): Capability[] {
  const out: Capability[] = [];
  if (stage === "distill") out.push("train.distill.dmd");
  else if (stage === "rl") out.push("train.llm.rl");
  else if (stage === "preference") out.push("train.llm.preference");
  else if (stage === "pretrain") out.push(kind === "llm" ? "train.llm.pretrain" : "train.video.full");
  else out.push(kind === "llm" ? "train.llm.sft" : adapter === "none" ? "train.video.full" : "train.video.lora");
  return out;
}