import type { ReactNode } from "react";
import { Icon, type IconName } from "./icons";

export interface EmptyStateProps {
  title: string;
  description?: string;
  icon?: IconName;
  action?: ReactNode;
}

export function EmptyState({ title, description, icon = "dashboard", action }: EmptyStateProps) {
  return (
    <div className="empty">
      <Icon name={icon} size={22} className="empty-icon" />
      <p className="empty-title">{title}</p>
      {description ? <p className="empty-desc">{description}</p> : null}
      {action ? <div className="empty-action">{action}</div> : null}
    </div>
  );
}

/** Inline error panel with a retry action; used by every data-backed page. */
export function ErrorState({
  title = "Could not load this panel",
  message,
  onRetry,
}: {
  title?: string;
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className="empty" role="alert">
      <Icon name="error" size={22} className="empty-icon" />
      <p className="empty-title">{title}</p>
      <p className="empty-desc">{message}</p>
      {onRetry ? (
        <div className="empty-action">
          <button type="button" className="btn btn-sm" onClick={onRetry}>
            <Icon name="refresh" size={14} />
            Retry
          </button>
        </div>
      ) : null}
    </div>
  );
}