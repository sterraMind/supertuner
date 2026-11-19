import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import type { MemoryReport, OOMRetry, RunEnd, RunEvent, RunStatus } from "@supertuner/core";
import { formatApiError, getApiClient, type StreamStatus } from "../lib/api";
import {
  copyText,
  useDocumentTitle,
  useNow,
  useResource,
} from "../lib/hooks";
import { elapsedSeconds, formatBytes, formatClock, formatDateTime, formatDuration, formatMib, formatNumber } from "../lib/format";
import { Badge, StatusBadge } from "../components/StatusBadge";
import { EmptyState, ErrorState } from "../components/EmptyState";
import { LossChart, type LossPoint } from "../components/LossChart";
import { StepTable, type StepRow } from "../components/StepTable";
import { SelectField } from "../components/Field";
import { useToast } from "../components/Toast";
import { Icon } from "../components/icons";

const LOG_LEVELS = ["all", "debug", "info", "warn", "error"] as const;
type LogLevelFilter = (typeof LOG_LEVELS)[number];

interface LogLine {
  ts: number;
  level: "debug" | "info" | "warn" | "error";
  message: string;
}

function eventKey(event: RunEvent): string {
  switch (event.type) {
    case "step":
      return `step:${event.step}`;
    case "eval":
      return `eval:${event.step}`;
    case "checkpoint":
      return `checkpoint:${event.step}`;
    case "log":
      return `log:${event.ts}:${event.level}:${event.message}`;
    case "error":
      return `error:${event.ts}:${event.message}`;
    case "phase_start":
      return `phase_start:${event.step_id}:${event.phase}`;
    case "phase_end":
      return `phase_end:${event.step_id}:${event.phase}`;
    case "run_start":
      return "run_start";
    case "run_end":
      return "run_end";
    case "memory_report":
      return `memory_report:${event.ts}`;
    case "oom_retry":
      return `oom_retry:${event.ts}:${event.attempt}`;
  }
}

/** Replay and live frames overlap; key-merge keeps one copy of each event. */
function mergeEvents(previous: readonly RunEvent[], incoming: readonly RunEvent[]): RunEvent[] {
  const merged = new Map<string, RunEvent>();
  for (const event of previous) merged.set(eventKey(event), event);
  for (const event of incoming) merged.set(eventKey(event), event);
  return [...merged.values()].sort((a, b) => a.ts - b.ts);
}

const STREAM_TONE: Record<StreamStatus, "ok" | "info" | "warn" | "neutral" | "error"> = {
  connecting: "info",
  open: "ok",
  reconnecting: "warn",
  closed: "neutral",
  failed: "error",
};

const STREAM_LABEL: Record<StreamStatus, string> = {
  connecting: "connecting",
  open: "live",
  reconnecting: "reconnecting",
  closed: "stream ended",
  failed: "stream failed",
};

export function RunDetail() {
  const { id = "" } = useParams<{ id: string }>();
  useDocumentTitle("Run monitor");
  const client = getApiClient();
  const toast = useToast();

  const runResource = useResource((signal) => client.getRun(id, signal), [id], { enabled: Boolean(id) });
  const reloadRef = useRef(runResource.reload);
  reloadRef.current = runResource.reload;

  const [events, setEvents] = useState<RunEvent[]>([]);
  const [streamStatus, setStreamStatus] = useState<StreamStatus>("connecting");
  const [streamError, setStreamError] = useState<string | null>(null);
  const [levelFilter, setLevelFilter] = useState<LogLevelFilter>("all");
  const [confirming, setConfirming] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [copied, setCopied] = useState(false);

  const run = runResource.data;

  // Replay the persisted event log so a reload shows history immediately.
  useEffect(() => {
    if (!id) return;
    setEvents([]);
    let live = true;
    client
      .getEvents(id)
      .then((list) => {
        if (live) setEvents((previous) => mergeEvents(previous, list));
      })
      .catch(() => {
        /* the stream will carry on from here; a missing replay is not fatal */
      });
    return () => {
      live = false;
    };
  }, [client, id]);

  const liveStatus = run?.status === "running" || run?.status === "queued";

  // Live SSE while the run can still produce frames.
  useEffect(() => {
    if (!id || !liveStatus) return;
    setStreamStatus("connecting");
    const stream = client.openRunStream(id, {
      onEvent: (event) => {
        setEvents((previous) => mergeEvents(previous, [event]));
        if (event.type === "run_end") reloadRef.current();
      },
      onStatus: (status) => setStreamStatus(status),
      onError: (message) => setStreamError(message),
    });
    return () => stream.close();
  }, [client, id, liveStatus]);

  const endEvent = useMemo(() => {
    const found = [...events].reverse().find((event) => event.type === "run_end");
    return found && found.type === "run_end" ? (found as RunEnd) : null;
  }, [events]);

  const status: RunStatus | null = endEvent ? endEvent.status : run?.status ?? null;
  const now = useNow(1000, status === "running" || status === "queued");
  const active = status === "running" || status === "queued";

  const steps = useMemo(() => events.filter((event): event is Extract<RunEvent, { type: "step" }> => event.type === "step"), [events]);
  const latestStep = steps[steps.length - 1];
  const lossPoints: LossPoint[] = steps.map((step) => ({ step: step.step, loss: step.loss }));
  const vramPeak = steps.reduce<number | null>((peak, step) => {
    if (step.vram_peak_mib === null) return peak;
    return peak === null ? step.vram_peak_mib : Math.max(peak, step.vram_peak_mib);
  }, null);
  const memoryReport = useMemo<MemoryReport | null>(() => {
    const found = [...events].reverse().find((event) => event.type === "memory_report");
    return found && found.type === "memory_report" ? (found as MemoryReport) : null;
  }, [events]);
  // OOM 有限次自动重试（v0.2 T4 / G4）：收集所有 oom_retry 事件（按 ts 升序），
  // 供下方显示「第 N 次重试换用更保守配置」的一行提示。
  const oomRetries = useMemo<OOMRetry[]>(() => events.filter((event): event is OOMRetry => event.type === "oom_retry"), [events]);
  const stepRows: StepRow[] = steps.map((step) => step);
  const maxSteps = run?.spec.train.max_steps ?? 0;
  const progress = status === "completed" ? 1 : maxSteps > 0 && latestStep ? Math.min(1, latestStep.step / maxSteps) : 0;

  const logLines = useMemo<LogLine[]>(() => {
    const lines: LogLine[] = [];
    for (const event of events) {
      if (event.type === "log") lines.push({ ts: event.ts, level: event.level, message: event.message });
      else if (event.type === "error") {
        lines.push({ ts: event.ts, level: "error", message: event.detail ? `${event.message} — ${event.detail}` : event.message });
      } else if (event.type === "checkpoint") {
        lines.push({ ts: event.ts, level: "info", message: `checkpoint at step ${event.step} → ${event.path}` });
      }
    }
    return lines;
  }, [events]);

  const visibleLogs = levelFilter === "all" ? logLines : logLines.filter((line) => line.level === levelFilter);

  const artifactsResource = useResource((signal) => client.getArtifacts(id, signal), [id], {
    enabled: Boolean(id) && status !== null && status !== "running" && status !== "queued",
  });
  const artifacts = artifactsResource.data ?? (run?.artifacts ?? []).map((path) => ({ path, bytes: 0, modified_at: "" }));

  const copyId = async (): Promise<void> => {
    const ok = await copyText(id);
    setCopied(ok);
    toast.push(ok ? "ok" : "warn", ok ? "Run id copied" : "Clipboard unavailable");
  };

  const cancelRun = async (): Promise<void> => {
    setCancelling(true);
    try {
      await client.cancelRun(id);
      toast.push("ok", "Cancellation requested", "The engine will stop after the current step.");
      setConfirming(false);
      reloadRef.current();
    } catch (error) {
      toast.push("error", "Cancel failed", formatApiError(error));
    } finally {
      setCancelling(false);
    }
  };

  if (runResource.status === "error") {
    return (
      <>
        <div className="page-head">
          <div className="page-head-copy">
            <span className="eyebrow">Run</span>
            <h1>Run unavailable</h1>
          </div>
          <div className="page-actions">
            <Link className="btn" to="/runs">
              <Icon name="chevron" size={13} />
              Back to runs
            </Link>
          </div>
        </div>
        <div className="panel">
          <div className="panel-body">
            <ErrorState title="Could not load this run" message={formatApiError(runResource.error)} onRetry={() => runResource.reload()} />
          </div>
        </div>
      </>
    );
  }

  if (!run && runResource.status === "loading") {
    return (
      <div className="panel">
        <div className="panel-body">
          <p className="small muted">Loading run…</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="monitor-head">
        <div className="monitor-title">
          <span className="eyebrow">Run monitor</span>
          <h1>{run?.name ?? id}</h1>
          <div className="row-wrap small muted">
            <span className="mono">{run?.id ?? id}</span>
            <span>·</span>
            <span className="mono">{run?.spec.config.family ?? "—"}</span>
            <span>·</span>
            <span>{run?.spec.config.stage ?? "—"}</span>
            <button type="button" className="btn btn-sm btn-ghost" onClick={() => void copyId()} aria-label="Copy run id">
              <Icon name={copied ? "check" : "copy"} size={12} />
            </button>
          </div>
        </div>
        <div className="page-actions">
          {status ? <StatusBadge status={status} /> : <Badge tone="neutral">unknown</Badge>}
          {active ? (
            <Badge tone={STREAM_TONE[streamStatus]}>{STREAM_LABEL[streamStatus]}</Badge>
          ) : (
            <Badge tone="neutral">finished</Badge>
          )}
          {active ? (
            <button type="button" className="btn btn-danger btn-sm" onClick={() => setConfirming(true)}>
              <Icon name="stop" size={13} />
              Cancel run
            </button>
          ) : null}
        </div>
      </div>

      <span className="sr-only" aria-live="polite">
        {status ? `Run status ${status}` : ""}
        {endEvent?.reason ? `. ${endEvent.reason}` : ""}
      </span>

      {streamError && active ? (
        <p className="small" role="status">
          <Icon name="warning" size={13} /> Live stream issue: {streamError}
        </p>
      ) : null}

      <div className="monitor-stats" aria-label="Live run statistics">
        <div className="monitor-stat">
          <span className="label">Step</span>
          <span className="value">{latestStep ? formatNumber(latestStep.step) : "—"}</span>
        </div>
        <div className="monitor-stat">
          <span className="label">Loss</span>
          <span className="value">{latestStep?.loss === null || latestStep?.loss === undefined ? "—" : formatNumber(latestStep.loss, 4)}</span>
        </div>
        <div className="monitor-stat">
          <span className="label">Tokens / s</span>
          <span className="value">{latestStep?.tokens_per_s === null || latestStep?.tokens_per_s === undefined ? "—" : formatNumber(latestStep.tokens_per_s, 1)}</span>
        </div>
        <div className="monitor-stat">
          <span className="label">VRAM peak</span>
          <span className="value">{formatMib(vramPeak ?? run?.headline_mib ?? null)}</span>
        </div>
        <div className="monitor-stat">
          <span className="label">Duration</span>
          <span className="value">
            {run ? formatDuration(elapsedSeconds(run.started_at, run.finished_at, now)) : "—"}
          </span>
        </div>
      </div>

      {confirming ? (
        <div className="confirm-panel" role="alertdialog" aria-labelledby="confirm-title" aria-describedby="confirm-body">
          <div className="stack-sm">
            <strong id="confirm-title">Cancel this run?</strong>
            <p id="confirm-body" className="small muted">
              The engine stops after the current step. Checkpoints already written stay on disk; the run is marked
              cancelled and cannot be resumed from the monitor.
            </p>
          </div>
          <div className="row">
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setConfirming(false)}>
              Keep running
            </button>
            <button type="button" className="btn btn-danger-solid btn-sm" disabled={cancelling} onClick={() => void cancelRun()}>
              <Icon name="stop" size={13} />
              {cancelling ? "Cancelling…" : "Yes, cancel"}
            </button>
          </div>
        </div>
      ) : null}

      <div className="grid-dash">
        <section className="panel" aria-labelledby="chart-heading">
          <div className="panel-head">
            <h2 id="chart-heading">Loss</h2>
            <span className="small faint">{steps.length} steps charted</span>
          </div>
          <div className="panel-body">
            <LossChart points={lossPoints} ariaLabel={`Loss curve for ${run?.name ?? id}`} idleLabel={active ? "Waiting for the first step" : "No loss data recorded"} />
            <div className="row spread small faint">
              <span>
                progress {formatNumber(progress * 100)}% of {formatNumber(maxSteps)} steps
              </span>
              <span className="mono">{latestStep?.lr === null || latestStep?.lr === undefined ? "—" : latestStep.lr.toExponential(2)} lr</span>
            </div>
          </div>
        </section>

        <div className="stack">
          <section className="panel" aria-labelledby="facts-heading">
            <div className="panel-head">
              <h2 id="facts-heading">Run facts</h2>
            </div>
            <div className="panel-body">
              <dl className="kv">
                <dt>status</dt>
                <dd>{status ?? "—"}</dd>
                <dt>created</dt>
                <dd>{run ? formatDateTime(run.created_at) : "—"}</dd>
                <dt>started</dt>
                <dd>{run ? formatDateTime(run.started_at) : "—"}</dd>
                <dt>finished</dt>
                <dd>{run ? formatDateTime(run.finished_at) : "—"}</dd>
                <dt>exit code</dt>
                <dd>{run?.exit_code === null || run?.exit_code === undefined ? "—" : run.exit_code}</dd>
                <dt>headline</dt>
                <dd>{formatMib(run?.headline_mib ?? run?.plan?.memory.training_peak_mib ?? null)}</dd>
                <dt>tokens / step</dt>
                <dd>{formatNumber(run?.tokens_per_step ?? run?.plan?.memory.tokens_per_step ?? null)}</dd>
                <dt>expected</dt>
                <dd>{run?.plan?.estimated_minutes === null || run?.plan?.estimated_minutes === undefined ? "—" : formatDuration(run.plan.estimated_minutes * 60)}</dd>
              </dl>
              {run?.error ? (
                <p className="small" role="alert">
                  <Icon name="error" size={13} /> {run.error}
                </p>
              ) : null}
              {endEvent?.reason ? <p className="small faint">end reason: {endEvent.reason}</p> : null}
            </div>
          </section>

          {memoryReport ? (
            <section className="panel" aria-labelledby="memory-heading">
              <div className="panel-head">
                <h2 id="memory-heading">Memory report</h2>
                <span className="small faint">
                  {memoryReport.stage ? `${memoryReport.stage} · ${memoryReport.adapter}` : ""}
                </span>
              </div>
              <div className="panel-body">
                <dl className="kv">
                  <dt>planned</dt>
                  <dd className="mono">{formatMib(memoryReport.planned_mib)}</dd>
                  <dt>measured peak</dt>
                  <dd className="mono">{formatMib(memoryReport.measured_peak_mib)}</dd>
                  <dt>delta</dt>
                  <dd className="mono">
                    {memoryReport.delta_mib === null
                      ? "—"
                      : `${memoryReport.delta_mib >= 0 ? "+" : ""}${formatMib(memoryReport.delta_mib)}`}
                  </dd>
                  <dt>vs plan</dt>
                  <dd className="mono">
                    {memoryReport.delta_pct === null
                      ? "—"
                      : `${memoryReport.delta_pct >= 0 ? "+" : ""}${memoryReport.delta_pct.toFixed(1)}%`}
                  </dd>
                </dl>
                {run?.plan?.memory?.calibration ? (
                  <p className="small">
                    已校准 {run.plan.memory.calibration.count} 次
                    {run.plan.memory.calibration.avg_delta_pct !== null
                      ? `（平均偏差 ${run.plan.memory.calibration.avg_delta_pct.toFixed(1)}%）`
                      : "（样本不足）"}
                  </p>
                ) : null}
              </div>
            </section>
          ) : null}

          {oomRetries.length > 0 ? (
            <section className="panel" aria-label="OOM retries">
            <div className="panel-body">
              {oomRetries.map((retry) => (
                <p className="small faint" key={retry.ts}>
                  OOM 重试第 {retry.attempt}/{retry.max_retries} 次（换块 → {retry.new_blocks_to_swap}，
                  量化 {retry.new_quant_base}，预算 {formatMib(retry.new_vram_budget_mib)}）
                </p>
              ))}
            </div>
            </section>
          ) : null}

          <section className="panel" aria-labelledby="artifacts-heading">
            <div className="panel-head">
              <h2 id="artifacts-heading">Artifacts</h2>
              <span className="small faint">{artifacts.length}</span>
            </div>
            <div className="panel-body">
              {artifacts.length === 0 ? (
                <p className="small faint">
                  {active ? "Artifacts appear when the run finishes." : "No artifacts recorded for this run."}
                </p>
              ) : (
                <ul className="artifacts">
                  {artifacts.map((artifact) => (
                    <li key={artifact.path} className="artifact">
                      <Icon name="folder" size={13} />
                      <span className="path mono">{artifact.path}</span>
                      {artifact.bytes > 0 ? (
                        <span className="small faint">{formatBytes(artifact.bytes)}</span>
                      ) : null}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>
        </div>
      </div>

      <section className="panel" aria-labelledby="steps-heading">
        <div className="panel-head">
          <h2 id="steps-heading">Steps</h2>
          <span className="small faint">newest first · latest 50</span>
        </div>
        <div className="panel-body">
          <StepTable steps={stepRows} emptyLabel={active ? "No steps received yet." : "This run recorded no steps."} />
        </div>
      </section>

      <section className="panel" aria-labelledby="logs-heading">
        <div className="panel-head">
          <h2 id="logs-heading">Logs</h2>
          <div className="row">
            <SelectField
              label="Level"
              value={levelFilter}
              options={LOG_LEVELS.map((level) => ({ value: level, label: level === "all" ? "All levels" : level }))}
              onChange={(level) => setLevelFilter(level as LogLevelFilter)}
            />
            <span className="small faint">{visibleLogs.length} lines</span>
          </div>
        </div>
        <div className="panel-body flush">
          {visibleLogs.length === 0 ? (
            <div className="panel-body">
              <EmptyState icon="runs" title="No log lines" description="Nothing at this level has been written yet." />
            </div>
          ) : (
            <div className="logs" role="log" aria-label="Run log" aria-live="off">
              {visibleLogs.slice(-400).map((line) => (
                <div key={`${line.ts}-${line.level}-${line.message}`} className="log-line">
                  <span className="log-time mono">{formatClock(line.ts)}</span>
                  <span className={`log-level ${line.level}`}>{line.level}</span>
                  <span className="log-message">{line.message}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>
    </>
  );
}