import type { PluginManifest } from "@supertuner/core";
import { formatApiError, getApiClient } from "../lib/api";
import { useDocumentTitle, useResource } from "../lib/hooks";
import { Badge } from "../components/StatusBadge";
import { EmptyState, ErrorState } from "../components/EmptyState";
import { Icon } from "../components/icons";

export function Plugins() {
  useDocumentTitle("Plugins");
  const client = getApiClient();
  const plugins = useResource((signal) => client.plugins(signal), []);

  const list = plugins.data?.plugins ?? [];
  const errors = plugins.data?.errors ?? [];
  const enabled = list.filter((plugin) => plugin.enabled).length;
  const experimental = list.filter((plugin) => plugin.experimental).length;

  return (
    <>
      <div className="page-head">
        <div className="page-head-copy">
          <span className="eyebrow">Plugins</span>
          <h1>Engine extensions</h1>
          <p>
            Model families, datasets, training strategies and exporters are discovered as Python entry points. A
            discovery failure never blocks the server — it is listed here with the reason.
          </p>
        </div>
        <div className="page-actions">
          <button type="button" className="btn btn-sm" onClick={() => plugins.reload()}>
            <Icon name="refresh" size={13} />
            Refresh
          </button>
        </div>
      </div>

      {plugins.data ? (
        <div className="row-wrap">
          <Badge tone="neutral">{list.length} loaded</Badge>
          <Badge tone="ok" icon="check">
            {enabled} enabled
          </Badge>
          {experimental > 0 ? <Badge tone="warn">{experimental} experimental</Badge> : null}
          {errors.length > 0 ? <Badge tone="error" icon="error">{errors.length} discovery errors</Badge> : null}
        </div>
      ) : null}

      {errors.length > 0 ? (
        <section className="panel" aria-labelledby="errors-heading">
          <div className="panel-head">
            <h2 id="errors-heading">Discovery errors</h2>
          </div>
          <div className="panel-body">
            <ul className="issues">
              {errors.map((error) => (
                <li key={`${error.entry_point}-${error.reason}`} className="issue issue-error">
                  <Icon name="error" size={15} />
                  <div className="issue-body">
                    <span className="issue-title mono">{error.entry_point || "(unknown entry point)"}</span>
                    <span className="issue-hint">{error.reason}</span>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </section>
      ) : null}

      {plugins.status === "error" ? (
        <div className="panel">
          <div className="panel-body">
            <ErrorState title="Plugin registry unavailable" message={formatApiError(plugins.error)} onRetry={() => plugins.reload()} />
          </div>
        </div>
      ) : plugins.status === "loading" && !plugins.data ? (
        <div className="panel">
          <div className="panel-body">
            <p className="small muted">Loading plugins…</p>
          </div>
        </div>
      ) : list.length === 0 ? (
        <EmptyState
          icon="plugins"
          title="No plugins loaded"
          description="Install a SuperTuner plugin package into the engine environment, then reload the engine from Settings."
        />
      ) : (
        <div className="plugin-list">
          {list.map((plugin) => (
            <PluginCard key={plugin.id} plugin={plugin} />
          ))}
        </div>
      )}
    </>
  );
}

function PluginCard({ plugin }: { plugin: PluginManifest }) {
  const homepageOk = /^https?:\/\//i.test(plugin.homepage);

  return (
    <article className="plugin" aria-labelledby={`plugin-${plugin.id}`}>
      <div className="spread">
        <div className="stack-sm">
          <h2 id={`plugin-${plugin.id}`} className="plugin-name">
            {plugin.name}
          </h2>
          <span className="plugin-version mono">v{plugin.version} · {plugin.id}</span>
        </div>
        <div className="row-wrap">
          <Badge tone="info">{plugin.kind}</Badge>
          {plugin.experimental ? <Badge tone="warn" icon="warning">experimental</Badge> : null}
          {plugin.enabled ? (
            <Badge tone="ok" icon="check">enabled</Badge>
          ) : (
            <Badge tone="neutral">disabled</Badge>
          )}
        </div>
      </div>

      <p className="plugin-desc">{plugin.description || "No description provided by the plugin."}</p>

      {plugin.capabilities.length > 0 ? (
        <div className="capabilities" aria-label="Capabilities">
          {plugin.capabilities.map((capability) => (
            <span key={capability} className="capability mono">
              {capability}
            </span>
          ))}
        </div>
      ) : (
        <p className="small faint">Declares no capabilities.</p>
      )}

      <dl className="kv">
        <dt>families</dt>
        <dd>{plugin.families.length > 0 ? plugin.families.join(", ") : "—"}</dd>
        <dt>licence</dt>
        <dd>{plugin.license || "unknown"}</dd>
        <dt>priority</dt>
        <dd className="mono">{plugin.priority}</dd>
        <dt>entry point</dt>
        <dd className="mono">{plugin.entry_point || "—"}</dd>
        <dt>requires</dt>
        <dd>{plugin.requires.length > 0 ? plugin.requires.join(", ") : "—"}</dd>
        <dt>homepage</dt>
        <dd>
          {homepageOk ? (
            <a href={plugin.homepage} target="_blank" rel="noreferrer noopener">
              {plugin.homepage}
            </a>
          ) : (
            "—"
          )}
        </dd>
      </dl>
    </article>
  );
}