import { useState } from "react";
import { apiUrl, formatApiError, getApiClient } from "../lib/api";
import { copyText, useDocumentTitle, useResource } from "../lib/hooks";
import { formatGib, formatMib } from "../lib/format";
import { Badge, HealthDot } from "../components/StatusBadge";
import { ErrorState } from "../components/EmptyState";
import { useToast } from "../components/Toast";
import { Icon, type IconName } from "../components/icons";
import { useTheme, type ThemePref } from "../theme/theme";

const THEME_CHOICES: { pref: ThemePref; label: string; description: string; icon: IconName }[] = [
  { pref: "system", label: "System", description: "Follow the OS appearance setting live.", icon: "monitor" },
  { pref: "light", label: "Light", description: "Warm paper surfaces for daylight work.", icon: "sun" },
  { pref: "dark", label: "Dark", description: "Graphite surfaces for long sessions.", icon: "moon" },
];

export function Settings() {
  useDocumentTitle("Settings");
  const client = getApiClient();
  const toast = useToast();
  const { pref, theme, setPref } = useTheme();
  const system = useResource((signal) => client.system(signal), []);
  const health = useResource((signal) => client.health(signal), []);
  const [reloading, setReloading] = useState(false);
  const [reloadResult, setReloadResult] = useState<string | null>(null);

  const copy = async (label: string, value: string): Promise<void> => {
    const ok = await copyText(value);
    toast.push(ok ? "ok" : "warn", ok ? `${label} copied` : "Clipboard unavailable");
  };

  const reloadEngine = async (): Promise<void> => {
    setReloading(true);
    setReloadResult(null);
    try {
      await client.reloadEngine();
      setReloadResult("Engine reloaded. Plugin discovery ran again.");
      toast.push("ok", "Engine reloaded");
      system.reload();
      health.reload();
    } catch (error) {
      const message = formatApiError(error);
      setReloadResult(message);
      toast.push("error", "Engine reload failed", message);
    } finally {
      setReloading(false);
    }
  };

  const base = client.baseUrl;

  return (
    <>
      <div className="page-head">
        <div className="page-head-copy">
          <span className="eyebrow">Settings</span>
          <h1>Workspace configuration</h1>
          <p>
            Paths and versions are read-only mirrors of the engine report. Appearance and the engine reload are the
            only controls this page writes.
          </p>
        </div>
      </div>

      <div className="grid-dash">
        <div className="stack">
          <section className="panel" aria-labelledby="appearance-heading">
            <div className="panel-head">
              <h2 id="appearance-heading">Appearance</h2>
              <Badge tone="neutral">resolved: {theme}</Badge>
            </div>
            <div className="panel-body">
              <div className="stack-sm" role="radiogroup" aria-label="Colour theme">
                {THEME_CHOICES.map((choice) => (
                  <button
                    key={choice.pref}
                    type="button"
                    role="radio"
                    aria-checked={pref === choice.pref}
                    aria-label={choice.label}
                    className="theme-choice"
                    onClick={() => setPref(choice.pref)}
                  >
                    <Icon name={choice.icon} size={18} />
                    <span className="stack-sm">
                      <span>{choice.label}</span>
                      <span className="small muted">{choice.description}</span>
                    </span>
                  </button>
                ))}
              </div>
              <p className="panel-note">
                Stored in localStorage under <span className="mono">supertuner.theme</span>. The inline script in
                index.html applies the stored value before the first paint, so there is no flash.
              </p>
            </div>
          </section>

          <section className="panel" aria-labelledby="paths-heading">
            <div className="panel-head">
              <h2 id="paths-heading">Paths</h2>
              <span className="small faint">read-only</span>
            </div>
            <div className="panel-body">
              {system.status === "error" ? (
                <ErrorState message={formatApiError(system.error)} onRetry={() => system.reload()} />
              ) : system.data ? (
                <div className="path-list">
                  {[
                    { label: "Home", value: system.data.home },
                    { label: "Runs", value: system.data.runs_dir },
                    { label: "Models", value: system.data.models_dir },
                    { label: "Datasets", value: system.data.datasets_dir },
                  ].map((path) => (
                    <div key={path.label} className="path">
                      <span className="path-label">{path.label}</span>
                      <span className="path-value mono" title={path.value}>
                        {path.value}
                      </span>
                      <button
                        type="button"
                        className="btn btn-icon btn-sm"
                        aria-label={`Copy ${path.label.toLowerCase()} path`}
                        onClick={() => void copy(path.label, path.value)}
                      >
                        <Icon name="copy" size={12} />
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="small muted">Reading the engine report…</p>
              )}
            </div>
          </section>

          <section className="panel" aria-labelledby="engine-heading">
            <div className="panel-head">
              <h2 id="engine-heading">Engine</h2>
              <Badge tone={health.data?.ok ? "ok" : "neutral"}>{health.data?.version ?? "unknown"}</Badge>
            </div>
            <div className="panel-body">
              {system.data ? (
                <dl className="kv">
                  <dt>engine version</dt>
                  <dd className="mono">{system.data.engine.version}</dd>
                  <dt>engine python</dt>
                  <dd className="mono">{system.data.engine.python ?? "—"}</dd>
                  <dt>engine path</dt>
                  <dd className="mono">{system.data.engine.path ?? "—"}</dd>
                  <dt>server version</dt>
                  <dd className="mono">{system.data.server.version}</dd>
                  <dt>runtime python</dt>
                  <dd className="mono">{system.data.env.python || "—"}</dd>
                  <dt>platform</dt>
                  <dd>{system.data.env.platform || "—"}</dd>
                  <dt>ffmpeg</dt>
                  <dd className="mono">{system.data.env.ffmpeg || "not found"}</dd>
                  <dt>system RAM</dt>
                  <dd className="mono">{system.data.env.ram_mib > 0 ? formatGib(system.data.env.ram_mib, 0) : "—"}</dd>
                  <dt>GPU</dt>
                  <dd>
                    {system.data.env.gpus.length === 0
                      ? "none reported"
                      : system.data.env.gpus
                          .map((gpu) => `${gpu.name} (${formatMib(gpu.vram_mib)})`)
                          .join(", ")}
                  </dd>
                </dl>
              ) : (
                <p className="small muted">Reading the engine report…</p>
              )}

              <div className="row-wrap">
                <button type="button" className="btn btn-sm" onClick={() => void reloadEngine()} disabled={reloading}>
                  <Icon name="refresh" size={13} />
                  {reloading ? "Reloading…" : "Reload engine"}
                </button>
                <span className="small faint">Re-runs entry-point discovery; running jobs are unaffected.</span>
              </div>
              {reloadResult ? (
                <p className="small" role="status">
                  {reloadResult}
                </p>
              ) : null}
            </div>
          </section>
        </div>

        <div className="stack">
          <section className="panel" aria-labelledby="connection-heading">
            <div className="panel-head">
              <h2 id="connection-heading">Connection</h2>
              <HealthDot ok={health.data?.ok === true} label={health.data?.ok ? "reachable" : "unreachable"} />
            </div>
            <div className="panel-body">
              <dl className="kv">
                <dt>api base</dt>
                <dd className="mono">{base || "same origin"}</dd>
                <dt>resolved</dt>
                <dd className="mono">{apiUrl("/health")}</dd>
                <dt>env override</dt>
                <dd className="mono">{import.meta.env.VITE_API_BASE || "not set"}</dd>
                <dt>server</dt>
                <dd className="mono">{health.data?.version ?? "—"}</dd>
              </dl>
              <p className="small faint">
                Set <span className="mono">VITE_API_BASE</span> at build time to point the bundle at another host. In
                development the Vite proxy forwards <span className="mono">/api</span> to the local server.
              </p>
              <button type="button" className="btn btn-sm" onClick={() => health.reload()}>
                <Icon name="refresh" size={13} />
                Check health
              </button>
            </div>
          </section>

          <section className="panel" aria-labelledby="packages-heading">
            <div className="panel-head">
              <h2 id="packages-heading">Python packages</h2>
            </div>
            <div className="panel-body">
              {system.data && system.data.env.packages.length > 0 ? (
                <ul className="stack-sm">
                  {system.data.env.packages.map((pkg) => (
                    <li key={pkg.name} className="row spread">
                      <span className="mono small">{pkg.name}</span>
                      <span className="row">
                        <span className="mono small faint">{pkg.version || "—"}</span>
                        <Badge tone={pkg.ok ? "ok" : "warn"} icon={pkg.ok ? "check" : "warning"}>
                          {pkg.ok ? "ok" : "missing"}
                        </Badge>
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="small faint">No package report yet.</p>
              )}
            </div>
          </section>
        </div>
      </div>
    </>
  );
}