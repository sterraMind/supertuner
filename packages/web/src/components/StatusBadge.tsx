import type { ReactNode } from "react";
import type { RunStatus } from "@supertuner/core";
import { Icon, type IconName } from "./icons";

export type BadgeTone = "neutral" | "ok" | "warn" | "error" | "info" | "accent";

export interface BadgeProps {
  tone?: BadgeTone;
  children: ReactNode;
  icon?: IconName;
  title?: string;
}

export function Badge({ tone = "neutral", children, icon, title }: BadgeProps) {
  return (
    <span className={`badge badge-${tone}`} title={title}>
      {icon ? <Icon name={icon} size={12} /> : null}
      {children}
    </span>
  );
}

const STATUS_LABEL: Record<RunStatus, string> = {
  queued: "queued",
  running: "running",
  completed: "completed",
  failed: "failed",
  cancelled: "cancelled",
};

export interface StatusBadgeProps {
  status: RunStatus;
  /** Adds the status word for screen readers when the dot alone is shown. */
  compact?: boolean;
}

export function StatusBadge({ status, compact }: StatusBadgeProps) {
  return (
    <span className={`badge status-badge status-${status}`}>
      <span className="status-dot" aria-hidden="true" />
      {compact ? <span className="sr-only">{STATUS_LABEL[status]}</span> : STATUS_LABEL[status]}
    </span>
  );
}

/** A single coloured dot for the sidebar / topbar health readout. */
export function HealthDot({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className="row" title={label}>
      <span className={`status-dot ${ok ? "ok" : "err"}`} aria-hidden="true" style={{ background: ok ? "var(--ok)" : "var(--err)" }} />
      <span className="sr-only">{label}</span>
    </span>
  );
}

export function FeasibilityBadge({ fits, detail }: { fits: boolean | null; detail: string }) {
  if (fits === null) {
    return (
      <Badge tone="neutral" icon="warning" title={detail}>
        VRAM unknown
      </Badge>
    );
  }
  return fits ? (
    <Badge tone="ok" icon="check" title={detail}>
      Fits
    </Badge>
  ) : (
    <Badge tone="error" icon="error" title={detail}>
      Needs more VRAM
    </Badge>
  );
}