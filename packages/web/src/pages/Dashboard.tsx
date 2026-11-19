import { Link } from "react-router-dom";
import { getApiClient } from "../lib/api";
import { formatGib, formatDuration, formatMib, formatRelative, elapsedSeconds } from "../lib/format";
import { useDocumentTitle, useResource } from "../lib/hooks";
import { Badge, StatusBadge } from "../components/StatusBadge";
import { EmptyState, ErrorState } from "../components/EmptyState";
import { Icon } from "../components/icons";
import { useToast } from "../components/Toast";
import { formatApiError } from "../lib/api";

export function Dashboard() {
  useDocumentTitle("Dashboard");
  const client = getApiClient();
  const toast = useToast();
  const system = useResource((signal) => client.system(signal), []);
  const runs = useResource((signal) => client.listRuns(6, signal), []);
  const health = useResource((signal) => client.health(signal), []);

  const env = system.data?.env;
  const gpu = env?.gpus[0];
  const pkgs = env?.packages ?? [];
  const okCount = pkgs.filter((pkg) => pkg.ok).length;
  const missing = pkgs.filter((pkg) => !pkg.ok);

  const reloadEngine = (): void => {
    toast.push("info", "Reloading engine");
    client
      .reloadEngine()
      .then(() => {
        toast.push("ok", "Engine reloaded");
        system.reload();
      })
      .catch((error: unknown) => toast.push("error", "Engine reload failed", formatApiError(error)));
  };

  return (
    <>
      <div className="page-head">
        <div className="page-head-copy">
          <span className="eyebrow">Control plane</span>
          <h1>Train on the card you already own</h1>
          <p>
            Plan a run against real memory limits, compile it into an execution sequence, then supervise every step
            from one instrument panel.
          </p>
        </div>
        <div className="page-actions">
          <Link className="btn btn-primary" to="/planner">
            <Icon name="play" size={15} />
            New run
          </Link>
          <Link className="btn" to="/recipes">
            <Icon name="recipes" size={15} />
            Browse recipes
          </Link>
        </div>
      </div>

      <div className="grid-dash">
        <section className="panel" aria-labelledby="system-heading">
          <div className="panel-head">
            <h2 id="system-heading">System</h2>
            <div className="row">
              {health.data ? (
                <Badge tone={health.data.ok ? "ok" : "error"} icon={health.data.ok ? "check" : "error"}>
                  {health.data.ok ? "online" : "degraded"}
                </Badge>
              ) : null}
              <button type="button" className="btn btn-sm btn-ghost" onClick={() => system.reload()}>
                <Icon name="refresh" size={13} />
                Refresh
              </button>
            </div>
          </div>

          {system.status === "error" ? (
            <div className="panel-body">
              <ErrorState
                title="System report unavailable"
                message={formatApiError(system.error)}
                onRetry={() => system.reload()}
              />
            </div>
          ) : system.status === "loading" && !system.data ? (
            <div className="panel-body">
              <p className="small muted">Reading hardware and packages…</p>
            </div>
          ) : system.data ? (
            <>
              <div className="panel-body">
                <div className="stat-row">
                  <div className="stat">
                    <span className="stat-label">GPU</span>
                    <span className="stat-value sm">{gpu ? gpu.name : "none"}</span>
                    <span className="stat-hint">
                      {gpu ? `index ${gpu.index} · driver ${gpu.driver || "unknown"}` : "no CUDA device reported"}
                    </span>
                  </div>
                  <div className="stat">
                    <span className="stat-label">VRAM</span>
                    <span className="stat-value">
                      {gpu ? formatGib(gpu.vram_total_mib || gpu.vram_mib, 0) : "—"}
                    </span>
                    <span className="stat-hint">
                      {gpu && gpu.vram_free_mib > 0
                        ? `${formatMib(gpu.vram_free_mib)} free of ${formatMib(gpu.vram_total_mib || gpu.vram_mib)}`
                        : gpu
                          ? formatMib(gpu.vram_mib)
                          : "unavailable"}
                    </span>
                  </div>
                  <div className="stat">
                    <span className="stat-label">System RAM</span>
                    <span className="stat-value">{env && env.ram_mib > 0 ? formatGib(env.ram_mib, 0) : "—"}</span>
                    <span className="stat-hint">reported by the engine</span>
                  </div>
                  <div className="stat">
                    <span className="stat-label">Disk</span>
                    <span className="stat-value">
                      {system.data.disk.total_mib > 0 ? formatGib(system.data.disk.free_mib, 0) : "—"}
                    </span>
                    <span className="stat-hint">
                      {system.data.disk.total_mib > 0
                        ? `free of ${formatGib(system.data.disk.total_mib, 0)} at ${system.data.disk.path}`
                        : "not reported by /api/system"}
                    </span>
                  </div>
                </div>

                <dl className="kv">
                  <dt>packages</dt>
                  <dd>
                    {pkgs.length === 0 ? "none reported" : `${okCount}/${pkgs.length} ready`}
                    {missing.length > 0 ? ` · ${missing.length} missing` : ""}
                  </dd>
                  <dt>python</dt>
                  <dd>{env?.python || "—"}</dd>
                  <dt>platform</dt>
                  <dd>{env?.platform || "—"}</dd>
                  <dt>ffmpeg</dt>
                  <dd>{env?.ffmpeg || "not found"}</dd>
                  <dt>engine</dt>
                  <dd>
                    {system.data.engine.version}
                    {system.data.engine.python ? ` · ${system.data.engine.python}` : ""}
                  </dd>
                  <dt>server</dt>
                  <dd>{system.data.server.version}</dd>
                </dl>

                {pkgs.length > 0 ? (
                  <div className="stack-sm">
                    <span className="eyebrow">Package inventory</span>
                    <div className="row-wrap">
                      {pkgs.map((pkg) => (
                        <Badge
                          key={pkg.name}
                          tone={pkg.ok ? "ok" : "warn"}
                          icon={pkg.ok ? "check" : "warning"}
                          title={pkg.hint || `${pkg.name} ${pkg.version}`}
                        >
                          {pkg.name} {pkg.version || ""}
                        </Badge>
                      ))}
                    </div>
                  </div>
                ) : null}

                {env && env.warnings.length > 0 ? (
                  <ul className="stack-sm small muted" aria-label="Environment warnings">
                    {env.warnings.map((warning) => (
                      <li key={warning} className="row">
                        <Icon name="warning" size={14} />
                        <span>{warning}</span>
                      </li>
                    ))}
                  </ul>
                ) : null}
              </div>
              <div className="panel-foot">
                <span className="small faint">engine {system.data.engine.version}</span>
                <button type="button" className="btn btn-sm" onClick={reloadEngine}>
                  <Icon name="refresh" size={13} />
                  Reload engine
                </button>
              </div>
            </>
          ) : (
            <div className="panel-body">
              <EmptyState title="No system report" description="The server has not answered /api/system yet." />
            </div>
          )}
        </section>

        <div className="stack">
          <section className="panel" aria-labelledby="actions-heading">
            <div className="panel-head">
              <h2 id="actions-heading">Quick actions</h2>
            </div>
            <div className="panel-body">
              <Link className="btn btn-block" to="/planner">
                <Icon name="planner" size={15} />
                Compose a RunSpec
              </Link>
              <Link className="btn btn-block" to="/recipes">
                <Icon name="recipes" size={15} />
                Pick a curated recipe
              </Link>
              <Link className="btn btn-block" to="/runs">
                <Icon name="runs" size={15} />
                Open the run monitor
              </Link>
              <Link className="btn btn-block" to="/plugins">
                <Icon name="plugins" size={15} />
                Inspect plugins
              </Link>
            </div>
          </section>

          <section className="panel" aria-labelledby="paths-heading">
            <div className="panel-head">
              <h2 id="paths-heading">Workspace</h2>
            </div>
            <div className="panel-body">
              {system.data ? (
                <dl className="kv">
                  <dt>home</dt>
                  <dd>{system.data.home}</dd>
                  <dt>runs</dt>
                  <dd>{system.data.runs_dir}</dd>
                  <dt>models</dt>
                  <dd>{system.data.models_dir}</dd>
                  <dt>datasets</dt>
                  <dd>{system.data.datasets_dir}</dd>
                </dl>
              ) : (
                <p className="small faint">Paths appear once the server answers.</p>
              )}
            </div>
          </section>
        </div>
      </div>

      <section className="panel" aria-labelledby="runs-heading">
        <div className="panel-head">
          <h2 id="runs-heading">Recent runs</h2>
          <Link className="btn btn-sm btn-ghost" to="/runs">
            All runs
            <Icon name="chevron" size={13} />
          </Link>
        </div>
        {runs.status === "error" ? (
          <div className="panel-body">
            <ErrorState message={formatApiError(runs.error)} onRetry={() => runs.reload()} />
          </div>
        ) : runs.status === "loading" && !runs.data ? (
          <div className="panel-body">
            <p className="small muted">Loading runs…</p>
          </div>
        ) : runs.data && runs.data.length > 0 ? (
          <div className="table-wrap">
            <table>
              <caption className="sr-only">Most recent runs</caption>
              <thead>
                <tr>
                  <th scope="col">Run</th>
                  <th scope="col">Status</th>
                  <th scope="col">Family</th>
                  <th scope="col">Stage</th>
                  <th scope="col">Created</th>
                  <th scope="col">Duration</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.map((run) => (
                  <tr key={run.id}>
                    <th scope="row">
                      <Link to={`/runs/${run.id}`}>{run.name}</Link>
                    </th>
                    <td>
                      <StatusBadge status={run.status} />
                    </td>
                    <td className="mono small">{run.spec.config.family}</td>
                    <td className="small">{run.spec.config.stage}</td>
                    <td className="small muted">{formatRelative(run.created_at)}</td>
                    <td className="mono small">
                      {formatDuration(elapsedSeconds(run.started_at, run.finished_at))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="panel-body">
            <EmptyState
              icon="runs"
              title="No runs yet"
              description="Nothing has been queued on this machine. Start from a recipe or build a spec in the planner."
              action={
                <Link className="btn btn-primary btn-sm" to="/planner">
                  <Icon name="play" size={13} />
                  Compose a run
                </Link>
              }
            />
          </div>
        )}
      </section>
    </>
  );
}