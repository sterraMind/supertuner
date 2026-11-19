import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { Icon, type IconName } from "./icons";
import { getApiClient, type HealthResponse } from "../lib/api";
import { useTheme, type ThemePref } from "../theme/theme";

interface NavItem {
  to: string;
  label: string;
  icon: IconName;
  end?: boolean;
}

const NAV: NavItem[] = [
  { to: "/", label: "Dashboard", icon: "dashboard", end: true },
  { to: "/recipes", label: "Recipes", icon: "recipes" },
  { to: "/planner", label: "Planner", icon: "planner" },
  { to: "/runs", label: "Runs", icon: "runs" },
  { to: "/plugins", label: "Plugins", icon: "plugins" },
  { to: "/settings", label: "Settings", icon: "settings" },
];

const TITLES: { match: RegExp; title: string }[] = [
  { match: /^\/runs\/[^/]+$/, title: "Run monitor" },
  { match: /^\/runs/, title: "Runs" },
  { match: /^\/recipes/, title: "Recipes" },
  { match: /^\/planner/, title: "Planner" },
  { match: /^\/plugins/, title: "Plugins" },
  { match: /^\/settings/, title: "Settings" },
  { match: /^\//, title: "Dashboard" },
];

function titleFor(pathname: string): string {
  return TITLES.find((entry) => entry.match.test(pathname))?.title ?? "SuperTuner";
}

const THEME_OPTIONS: { pref: ThemePref; label: string; icon: IconName }[] = [
  { pref: "system", label: "System theme", icon: "monitor" },
  { pref: "light", label: "Light theme", icon: "sun" },
  { pref: "dark", label: "Dark theme", icon: "moon" },
];

export function ThemeSwitch() {
  const { pref, setPref } = useTheme();
  return (
    <div className="theme-switch" role="group" aria-label="Colour theme">
      {THEME_OPTIONS.map((option) => (
        <button
          key={option.pref}
          type="button"
          aria-pressed={pref === option.pref}
          aria-label={option.label}
          title={option.label}
          onClick={() => setPref(option.pref)}
        >
          <Icon name={option.icon} size={14} />
        </button>
      ))}
    </div>
  );
}

/** Polls /api/health so the shell can show whether the server is reachable. */
function useHealth(): { health: HealthResponse | null; online: boolean } {
  const [state, setState] = useState<{ health: HealthResponse | null; online: boolean }>({
    health: null,
    online: false,
  });

  useEffect(() => {
    let cancelled = false;
    const client = getApiClient();
    const check = (): void => {
      client
        .health()
        .then((health) => {
          if (!cancelled) setState({ health, online: health.ok });
        })
        .catch(() => {
          if (!cancelled) setState((previous) => ({ health: previous.health, online: false }));
        });
    };
    check();
    const timer = setInterval(check, 20_000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  return state;
}

export function Layout() {
  const location = useLocation();
  const { health, online } = useHealth();

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">
        Skip to content
      </a>

      <aside className="sidebar">
        <NavLink to="/" className="brand" aria-label="SuperTuner home">
          <span className="brand-mark">SuperTuner</span>
          {health?.version ? <span className="brand-version">{health.version}</span> : null}
        </NavLink>

        <nav className="nav" aria-label="Primary">
          <span className="nav-label eyebrow">Workspace</span>
          {NAV.map((item) => (
            <NavLink key={item.to} to={item.to} className="nav-link" end={item.end}>
              <Icon name={item.icon} size={18} />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-foot">
          <span className="kicker row">
            <span
              className="status-dot"
              aria-hidden="true"
              style={{ background: online ? "var(--ok)" : "var(--err)" }}
            />
            {online ? "server reachable" : "server offline"}
          </span>
          <span className="kicker">
            {health?.version ? `api ${health.version}` : "api unreachable"}
          </span>
          <span className="sr-only" role="status">
            {online ? "Server reachable" : "Server offline"}
          </span>
        </div>
      </aside>

      <div className="main-col">
        <header className="topbar">
          <span className="topbar-title">{titleFor(location.pathname)}</span>
          <div className="topbar-meta">
            <span className="row small muted">
              <Icon name="gpu" size={15} />
              single-GPU studio
            </span>
            <ThemeSwitch />
          </div>
        </header>
        <main id="main" className="page">
          <Outlet />
        </main>
      </div>
    </div>
  );
}