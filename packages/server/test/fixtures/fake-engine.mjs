#!/usr/bin/env node
/**
 * Fake `supertuner-engine` for server tests.
 *
 * Behaviour is driven by the spec's `config.name` prefix so parallel tests can
 * differ without sharing environment variables:
 *   infeasible*  compile returns feasible:false + error issues, exit 2
 *   crash*       compile exits 3 with a stderr trace
 *   badjson*     compile prints non-JSON and exits 0
 *   refuse*      compile returns {ok:false}, exit 0
 *   slow*        run emits 12 steps ~150 ms apart (cancel window)
 *   fail-run*    run emits an error event and run_end failed, exit 3
 *   crash-run*   run exits 3 without run_end
 *   anything     run emits 3 steps ~20 ms apart and completes
 *
 * Cancellation: `run` polls `<run-dir>/cancel`, emits run_end cancelled and
 * exits 130, mirroring the real engine contract.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const argv = process.argv.slice(2);
const command = argv[0];

function flag(name) {
  const index = argv.indexOf(name);
  return index >= 0 ? argv[index + 1] : undefined;
}

function has(name) {
  return argv.includes(name);
}

function emit(value) {
  process.stdout.write(`${JSON.stringify(value)}\n`);
}

function readSpec() {
  const path = flag("--spec");
  if (!path) {
    process.stderr.write("fake-engine: missing --spec\n");
    process.exit(3);
  }
  return JSON.parse(readFileSync(path, "utf8"));
}

function issue(severity, code, message, field = null) {
  return { severity, code, message, hint: "fake engine hint", field };
}

function buildPlan(spec, feasible) {
  const config = spec.config ?? {};
  const budget = config.vram_budget_mib ?? 8192;
  const ram = config.ram_budget_mib ?? 32768;
  const trainingPeak = feasible ? Math.floor(budget * 0.7) : budget + 2048;
  const samplingPeak = feasible ? Math.floor(budget * 0.4) : budget + 1024;
  const headline = Math.max(trainingPeak, samplingPeak);
  const issues = feasible
    ? []
    : [issue("error", "vram_exceeded", `fake plan needs ${trainingPeak} MiB but only ${budget} MiB are budgeted`, "config.vram_budget_mib")];
  return {
    feasible,
    memory: {
      feasible,
      vram_budget_mib: budget,
      ram_budget_mib: ram,
      training_peak_mib: trainingPeak,
      sampling_peak_mib: samplingPeak,
      headline_mib: headline,
      headroom_mib: budget - headline,
      ram_peak_mib: Math.floor(ram * 0.5),
      ram_headroom_mib: Math.floor(ram * 0.5),
      lines: [
        { label: "weights", mib: Math.floor(budget * 0.35), note: "fake" },
        { label: "optimizer", mib: Math.floor(budget * 0.2), note: "" },
      ],
      risks: feasible ? [] : ["fake infeasible plan"],
      tokens_per_step: 2048,
      notes: ["fake engine plan"],
    },
    execution: {
      backend: `fake-${config.family ?? "unknown"}`,
      backend_version: "0.0.0-fake",
      steps: [
        {
          id: "train",
          kind: "train",
          title: "Train (fake)",
          description: "no-op step emitted by the fake engine",
          command: [process.execPath, "-e", "process.exit(0)"],
          cwd: null,
          env: {},
          optional: false,
          estimated_mib: trainingPeak,
        },
      ],
      issues,
      notes: [],
      expected_artifacts: ["output/adapter.safetensors"],
    },
    issues,
    estimated_minutes: 5,
  };
}

function now() {
  return Date.now();
}

if (command === "version") {
  emit({ version: "0.0.0-fake" });
  process.exit(0);
}

if (command === "plugins") {
  emit({
    plugins: [
      {
        id: "fake.plugin",
        name: "Fake Engine Plugin",
        version: "0.0.1",
        kind: "backend",
        description: "fixture plugin",
        capabilities: [],
        families: ["fake-llm"],
        entry_point: "fake_engine.plugin:register",
        requires: [],
        license: "MIT",
        homepage: "",
        enabled: true,
        experimental: false,
        priority: 10,
      },
    ],
    errors: [],
  });
  process.exit(0);
}

if (command === "env") {
  emit({
    python: "3.12.0 (fake)",
    platform: "fixture-1.0",
    packages: [
      { name: "torch", version: "2.5.0", ok: true, hint: "" },
      { name: "transformers", version: null, ok: false, hint: "not installed" },
    ],
    gpus: [],
    ram_mib: 32768,
    ffmpeg: null,
    warnings: ["fake engine environment"],
  });
  process.exit(0);
}

if (command === "families") {
  emit({
    llm: [
      {
        key: "fake-llm",
        kind: "llm",
        params_b: 1.0,
        hidden: 2048,
        layers: 24,
        heads: 16,
        kv_heads: 16,
        head_dim: 128,
        ffn: 8192,
        vocab: 49152,
      },
    ],
    video: [
      {
        key: "fake-video",
        kind: "video",
        params_b: 1.3,
        hidden: 1536,
        layers: 30,
        heads: 12,
        head_dim: 128,
        ffn: 8960,
        patch: [1, 2, 2],
        vae_spatial_stride: 16,
        latent_channels: 16,
        fps: 16,
        supports_block_swap: true,
      },
    ],
  });
  process.exit(0);
}

if (command === "compile") {
  const spec = readSpec();
  const name = spec.config?.name ?? "";
  if (name.startsWith("crash") && !name.startsWith("crash-run")) {
    process.stderr.write("Traceback (most recent call last):\n  fake compile crash\n");
    process.exit(3);
  }
  if (name.startsWith("badjson")) {
    process.stdout.write("this is not JSON\n");
    process.exit(0);
  }
  if (name.startsWith("refuse") && !name.startsWith("refuse-run")) {
    emit({ ok: false, error: "fake engine refused the spec: unknown family", issues: [issue("error", "unknown_family", "family not in the fake registry", "config.family")] });
    process.exit(0);
  }
  const feasible = !name.startsWith("infeasible");
  const plan = buildPlan(spec, feasible);
  emit({ ok: true, plan });
  process.exit(feasible ? 0 : 2);
}

if (command === "run") {
  const spec = readSpec();
  const name = spec.config?.name ?? "";
  const runDir = flag("--run-dir");
  if (!runDir) {
    process.stderr.write("fake-engine: missing --run-dir\n");
    process.exit(3);
  }
  if (!has("--events-stdout")) {
    process.stderr.write("fake-engine: expected --events-stdout\n");
    process.exit(3);
  }
  mkdirSync(join(runDir, "output"), { recursive: true });
  mkdirSync(join(runDir, "logs"), { recursive: true });
  const runId = name || "fake-run";
  emit({ ts: now(), type: "run_start", run_id: runId, plan: null, resumed_from_step: null });

  if (name.startsWith("noisy")) {
    // One malformed line and engine events this server must preserve verbatim.
    process.stdout.write("this line is not an event\n");
    emit({ ts: now(), type: "phase_start", step_id: "train", phase: "train", title: "Training" });
    emit({ ts: now(), type: "phase_end", step_id: "train", phase: "train", status: "completed", exit_code: 0, reason: null });
  }

  if (name.startsWith("crash-run")) {
    process.stderr.write("Traceback (most recent call last):\n  fake training crash\n");
    process.exit(3);
  }

  if (name.startsWith("refuse-run")) {
    // A warning-less infeasible plan: engine emits error + run_end, exits 2.
    emit({ ts: now(), type: "error", message: "fake plan is infeasible without warnings", detail: null });
    emit({ ts: now(), type: "run_end", status: "failed", steps: 0, reason: "plan is infeasible on this budget" });
    process.exit(2);
  }

  const slow = name.startsWith("slow");
  const totalSteps = slow ? 12 : 3;
  const delayMs = slow ? 150 : 20;
  let step = 0;

  const timer = setInterval(() => {
    if (existsSync(join(runDir, "cancel"))) {
      emit({ ts: now(), type: "run_end", status: "cancelled", steps: step, reason: "cancelled by operator" });
      process.exit(130);
    }
    step += 1;
    emit({
      ts: now(),
      type: "step",
      step,
      loss: Math.max(0.1, 2.0 - step * 0.05),
      lr: 0.0001,
      grad_norm: 0.9,
      tokens_per_s: 1200 + step,
      vram_peak_mib: 4096,
      elapsed_s: step * 0.02,
    });
    if (name.startsWith("fail-run") && step === 2) {
      emit({ ts: now(), type: "error", message: "fake dataset vanished", detail: null });
      emit({ ts: now(), type: "run_end", status: "failed", steps: step, reason: "fake dataset vanished" });
      process.exit(3);
    }
    if (step >= totalSteps) {
      const artifact = join(runDir, "output", "adapter.safetensors");
      writeFileSync(artifact, "fake adapter weights");
      emit({ ts: now(), type: "checkpoint", step, path: artifact, bytes: 19 });
      emit({ ts: now(), type: "run_end", status: "completed", steps: step, reason: null });
      clearInterval(timer);
      process.exit(0);
    }
  }, delayMs);
  timer.unref();
  continueRun();

  // Keep the process alive while the interval drives it.
  function continueRun() {
    setInterval(() => undefined, 1_000).unref();
    process.on("SIGTERM", () => {
      emit({ ts: now(), type: "run_end", status: "cancelled", steps: step, reason: "SIGTERM" });
      process.exit(130);
    });
    setTimeout(() => undefined, 60_000);
  }
} else if (command !== undefined) {
  process.stderr.write(`fake-engine: unknown command ${command}\n`);
  process.exit(2);
}