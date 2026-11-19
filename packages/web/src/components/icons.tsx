import type { ReactNode } from "react";

/**
 * Hand-drawn icon set. One 24px grid, 1.75px stroke, round caps and joins.
 * Everything inherits `currentColor`; nothing here is an emoji or an image.
 */

export type IconName =
  | "dashboard"
  | "recipes"
  | "planner"
  | "runs"
  | "plugins"
  | "settings"
  | "play"
  | "stop"
  | "gpu"
  | "memory"
  | "disk"
  | "check"
  | "warning"
  | "error"
  | "info"
  | "sun"
  | "moon"
  | "monitor"
  | "chevron"
  | "copy"
  | "folder"
  | "download"
  | "refresh";

const paths: Record<IconName, ReactNode> = {
  dashboard: (
    <>
      <rect x="3.5" y="3.5" width="7" height="7" rx="1.5" />
      <rect x="13.5" y="3.5" width="7" height="7" rx="1.5" />
      <rect x="3.5" y="13.5" width="7" height="7" rx="1.5" />
      <rect x="13.5" y="13.5" width="7" height="7" rx="1.5" />
    </>
  ),
  recipes: (
    <>
      <rect x="3.5" y="4.5" width="17" height="6.5" rx="1.5" />
      <rect x="3.5" y="13" width="17" height="6.5" rx="1.5" />
      <path d="M6.75 7.75h6M6.75 16.25h6" />
      <path d="M17 7.75h.5M17 16.25h.5" />
    </>
  ),
  planner: (
    <>
      <path d="M4 7h16M4 12h16M4 17h16" />
      <circle cx="9" cy="7" r="2" fill="var(--bg-elevated)" />
      <circle cx="15" cy="12" r="2" fill="var(--bg-elevated)" />
      <circle cx="7" cy="17" r="2" fill="var(--bg-elevated)" />
    </>
  ),
  runs: (
    <>
      <path d="M3 12h3.5L9 5.5l4.5 13L16 12h5" />
    </>
  ),
  plugins: (
    <>
      <rect x="4.5" y="4.5" width="15" height="15" rx="3" />
      <circle cx="12" cy="12" r="3.2" />
      <path d="M12 8.8V6M12 18v-2.8M15.2 12H18M6 12h2.8" />
    </>
  ),
  settings: (
    <>
      <circle cx="12" cy="12" r="3.25" />
      <path d="M12 3v2.2M12 18.8V21M3 12h2.2M18.8 12H21M5.6 5.6l1.6 1.6M16.8 16.8l1.6 1.6M18.4 5.6l-1.6 1.6M7.2 16.8l-1.6 1.6" />
    </>
  ),
  play: <path d="M9 5.5l9.5 6.5-9.5 6.5z" />,
  stop: <rect x="7.5" y="7.5" width="9" height="9" rx="2" />,
  gpu: (
    <>
      <rect x="7" y="7" width="10" height="10" rx="2" />
      <rect x="10.25" y="10.25" width="3.5" height="3.5" rx="0.9" />
      <path d="M9.5 3.5v3M12 3.5v3M14.5 3.5v3M9.5 17.5v3M12 17.5v3M14.5 17.5v3M3.5 9.5h3M3.5 12h3M3.5 14.5h3M17.5 9.5h3M17.5 12h3M17.5 14.5h3" />
    </>
  ),
  memory: (
    <>
      <rect x="3.5" y="8" width="17" height="8" rx="1.5" />
      <path d="M7 11v2M11 11v2M15 11v2" />
      <path d="M6.5 16v2.5M9.5 16v2.5M12.5 16v2.5M15.5 16v2.5M18.5 16v2.5" />
    </>
  ),
  disk: (
    <>
      <rect x="3.5" y="5.5" width="17" height="13" rx="2.5" />
      <circle cx="16.5" cy="12" r="1.3" />
      <path d="M7 9.5h5.5" />
    </>
  ),
  check: <path d="M5.5 12.5l4.2 4.2L18.5 7.5" />,
  warning: (
    <>
      <path d="M12 4.25l8.75 15.25H3.25z" />
      <path d="M12 9v4.5" />
      <path d="M12 16.4h.01" />
    </>
  ),
  error: (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M9 9l6 6M15 9l-6 6" />
    </>
  ),
  info: (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 11v5.5" />
      <path d="M12 7.9h.01" />
    </>
  ),
  sun: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2.75v2.1M12 19.15v2.1M2.75 12h2.1M19.15 12h2.1M5.45 5.45l1.5 1.5M17.05 17.05l1.5 1.5M18.55 5.45l-1.5 1.5M6.95 17.05l-1.5 1.5" />
    </>
  ),
  moon: <path d="M20 14.6A8.6 8.6 0 0 1 9.4 4a8.6 8.6 0 1 0 10.6 10.6z" />,
  monitor: (
    <>
      <rect x="3" y="4.5" width="18" height="12" rx="2" />
      <path d="M9 20h6M12 16.5V20" />
    </>
  ),
  chevron: <path d="M9 6l6 6-6 6" />,
  copy: (
    <>
      <rect x="9" y="9" width="11" height="11" rx="2" />
      <path d="M15 9V6.5A2.5 2.5 0 0 0 12.5 4h-6A2.5 2.5 0 0 0 4 6.5v6A2.5 2.5 0 0 0 6.5 15H9" />
    </>
  ),
  folder: (
    <path d="M3.5 7.5a2 2 0 0 1 2-2h3.2l2 2.5h7.8a2 2 0 0 1 2 2v7.5a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z" />
  ),
  download: (
    <>
      <path d="M12 4v10.5" />
      <path d="M7.5 10.5l4.5 4 4.5-4" />
      <path d="M5 19.5h14" />
    </>
  ),
  refresh: (
    <>
      <path d="M19.8 12a7.8 7.8 0 1 1-2.3-5.5" />
      <path d="M20 4v5h-5" />
    </>
  ),
};

export interface IconProps {
  name: IconName;
  size?: number;
  /** Accessible name. Omit when the icon is decorative next to real text. */
  title?: string;
  className?: string;
  strokeWidth?: number;
}

export function Icon({ name, size = 20, title, className, strokeWidth = 1.75 }: IconProps) {
  return (
    <svg
      className={className ? `icon ${className}` : "icon"}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      role={title ? "img" : undefined}
      aria-hidden={title ? undefined : true}
      focusable="false"
    >
      {title ? <title>{title}</title> : null}
      {paths[name]}
    </svg>
  );
}

export const ICON_NAMES = Object.keys(paths) as IconName[];