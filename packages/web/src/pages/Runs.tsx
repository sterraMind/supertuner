import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import type { RunRecord, RunStatus } from "@supertuner/core";
import { formatApiError, getApiClient } from "../lib/api";
import { formatDuration, formatNumber, formatPercent, formatRelative, elapsedSeconds } from "../lib/format";
import { useDocumentTitle, useNow, useResource } from "../lib/hooks";
import { StatusBadge } from "../components/StatusBadge";
import { EmptyState, ErrorState } from "../components/EmptyState";
import { SelectField, TextField } from "../components/Field";
import { Icon } from "../components/icons";

const STATUS_FILTERS: { value: "all" | RunStatus; label: string }[] = [
  { value: "all", label: "All" },
  { value: "queued", label: "Queued" },
  { value: "running", label: "Running" },
  { value: "completed", label: "Completed" },
  { value: "failed", label: "Failed" },
  { value: "cancelled", label: "Cancelled" },
];

export function Runs() {
  useDocumentTitle("Runs");
  const client = getApiClient();
  const [status, setStatus] = useState<"all" | RunStatus>("all");
  const [search, setSearch] = useState("");
  const [limit, setLimit] = useState(50);

  const runs = useResource((signal) => client.listRuns(limit, signal), [limit]);

  const filtered = useMemo(() => {
    const list = runs.data ?? [];
    const needle = search.trim().toLowerCase();
    return list.filter((run) => {
      if (status !== "all" && run.status !== status) return false;
      if (!needle) return true;
      return (
        run.name.toLowerCase().includes(needle) ||
        run.id.toLowerCase().includes(needle) ||
        run.spec.config.family.toLowerCase().includes(needle) ||
        run.spec.config.stage.toLowerCase().includes(needle)
      );
    });
  }, [runs.data, search, status]);

  const total = runs.data?.length ?? 0;

  return (
    <>
      <div className="page-head">
        <div className="page-head-copy">
          <span className="eyebrow">Runs</span>
          <h1>Run history</h1>
          <p>Every run the engine has queued on this machine, with live telemetry for the ones still training.</p>
        </div>
        <div className="page-actions">
          <Link className="btn btn-primary" to="/planner">
            <Icon name="play" size={15} />
            New run
          </Link>
        </div>
      </div>

      <div className="filter-bar">
        <div className="segmented" role="group" aria-label="Filter by status">
          {STATUS_FILTERS.map((filter) => (
            <button
              key={filter.value}
              type="button"
              aria-pressed={status === filter.value}
              onClick={() => setStatus(filter.value)}
            >
              {filter.label}
            </button>
          ))}
        </div>
        <div className="filter-actions">
          <TextField
            label="Search"
            value={search}
            placeholder="name, id, family, stage"
            onChange={setSearch}
          />
          <SelectField
            label="Limit"
            value={String(limit)}
            options={[
              { value: "25", label: "25" },
              { value: "50", label: "50" },
              { value: "100", label: "100" },
            ]}
            onChange={(value) => setLimit(Number(value))}
          />
        </div>
      </div>

      <section className="panel" aria-labelledby="runs-heading">
        <div className="panel-head">
          <h2 id="runs-heading">
            {filtered.length} of {total} run{total === 1 ? "" : "s"}
          </h2>
          <button type="button" className="btn btn-sm btn-ghost" onClick={() => runs.reload()}>
            <Icon name="refresh" size={13} />
            Refresh
          </button>
        </div>

        {runs.status === "error" ? (
          <div className="panel-body">
            <ErrorState title="Run list unavailable" message={formatApiError(runs.error)} onRetry={() => runs.reload()} />
          </div>
        ) : runs.status === "loading" && !runs.data ? (
          <div className="panel-body">
            <p className="small muted">Loading runs…</p>
          </div>
        ) : filtered.length === 0 ? (
          <div className="panel-body">
            {total === 0 ? (
              <EmptyState
                icon="runs"
                title="No runs recorded"
                description="Nothing has been queued yet. Compose a spec in the planner and the first run lands here."
                action={
                  <Link className="btn btn-primary btn-sm" to="/planner">
                    <Icon name="play" size={13} />
                    Compose a run
                  </Link>
                }
              />
            ) : (
              <EmptyState
                icon="runs"
                title="No run matches those filters"
                description="Clear the search box or widen the status filter."
                action={
                  <button
                    type="button"
                    className="btn btn-sm"
                    onClick={() => {
                      setSearch("");
                      setStatus("all");
                    }}
                  >
                    Clear filters
                  </button>
                }
              />
            )}
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <caption className="sr-only">Runs, newest first</caption>
              <thead>
                <tr>
                  <th scope="col">Run</th>
                  <th scope="col">Status</th>
                  <th scope="col">Family / stage</th>
                  <th scope="col">Progress</th>
                  <th scope="col">Loss</th>
                  <th scope="col">Tok/s</th>
                  <th scope="col">Duration</th>
                  <th scope="col">Created</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((run) => (
                  <RunRow key={run.id} run={run} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}

interface Telemetry {
  step: number | null;
  loss: number | null;
  tokensPerSecond: number | null;
}

function useRunTelemetry(run: RunRecord): Telemetry {
  const client = getApiClient();
  const live = run.status === "running";
  const events = useResource((signal) => client.getEvents(run.id, signal), [run.id, run.status], {
    enabled: live,
  });

  return useMemo<Telemetry>(() => {
    if (!live) {
      return { step: null, loss: null, tokensPerSecond: run.tokens_per_step };
    }
    const steps = (events.data ?? []).filter((event) => event.type === "step");
    const last = steps[steps.length - 1];
    if (!last || last.type !== "step") return { step: null, loss: null, tokensPerSecond: run.tokens_per_step };
    return { step: last.step, loss: last.loss, tokensPerSecond: last.tokens_per_s };
  }, [events.data, live, run.tokens_per_step]);
}

function RunRow({ run }: { run: RunRecord }) {
  const now = useNow(1000, run.status === "running");
  const telemetry = useRunTelemetry(run);
  const maxSteps = run.spec.train.max_steps;
  const fraction =
    run.status === "completed"
      ? 1
      : run.status === "cancelled" || run.status === "failed"
        ? null
        : telemetry.step !== null && maxSteps > 0
          ? Math.min(1, telemetry.step / maxSteps)
          : 0;

  const duration = elapsedSeconds(run.started_at, run.finished_at, now);

  return (
    <tr>
      <th scope="row">
        <Link to={`/runs/${encodeURIComponent(run.id)}`}>{run.name}</Link>
        <div className="xs faint mono">{run.id}</div>
      </th>
      <td>
        <StatusBadge status={run.status} />
      </td>
      <td>
        <span className="mono small">{run.spec.config.family}</span>
        <div className="xs faint">{run.spec.config.stage}</div>
      </td>
      <td style={{ minWidth: 140 }}>
        {fraction === null ? (
          <span className="small faint">stopped at {telemetry.step === null ? "—" : formatNumber(telemetry.step)}</span>
        ) : (
          <>
            <div
              className="progress"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(fraction * 100)}
              aria-label={`${run.name} progress`}
            >
              <div className="progress-fill" style={{ width: formatPercent(fraction * 100, 0) }} />
            </div>
            <span className="progress-meta small faint">
              {telemetry.step === null ? "—" : formatNumber(telemetry.step)} / {formatNumber(maxSteps)} ·{" "}
              {formatPercent(fraction * 100, 0)}
            </span>
          </>
        )}
      </td>
      <td className="mono small">{telemetry.loss === null ? "—" : formatNumber(telemetry.loss, 4)}</td>
      <td className="mono small">
        {telemetry.tokensPerSecond === null ? "—" : formatNumber(telemetry.tokensPerSecond, 1)}
      </td>
      <td className="mono small">{formatDuration(duration)}</td>
      <td className="small muted">{formatRelative(run.created_at)}</td>
    </tr>
  );
}