import type { Issue } from "@supertuner/core";
import { Icon, type IconName } from "./icons";

const ICONS: Record<Issue["severity"], IconName> = {
  error: "error",
  warning: "warning",
  info: "info",
};

const TITLES: Record<Issue["severity"], string> = {
  error: "Error",
  warning: "Warning",
  info: "Note",
};

export interface IssueListProps {
  issues: readonly Issue[];
  /** Called with the `field` path when the operator clicks an issue. */
  onFocusField?: (field: string) => void;
  emptyLabel?: string | null;
}

export function IssueList({ issues, onFocusField, emptyLabel = "No issues reported." }: IssueListProps) {
  if (issues.length === 0) {
    if (!emptyLabel) return null;
    return (
      <p className="small faint row">
        <Icon name="check" size={14} />
        {emptyLabel}
      </p>
    );
  }

  return (
    <ul className="issues">
      {issues.map((issue, index) => {
        const clickable = Boolean(issue.field && onFocusField);
        const content = (
          <>
            <Icon name={ICONS[issue.severity]} size={15} />
            <div className="issue-body">
              <p className="issue-title">
                {issue.message}
                <span className="issue-meta">
                  {issue.code}
                  {issue.field ? ` · ${issue.field}` : ""}
                </span>
              </p>
              {issue.hint ? <p className="issue-hint">{issue.hint}</p> : null}
            </div>
          </>
        );
        return (
          <li
            key={`${issue.code}-${issue.field ?? "none"}-${index}`}
            className={`issue issue-${issue.severity}`}
            aria-label={`${TITLES[issue.severity]}: ${issue.message}`}
          >
            {clickable ? (
              <button
                type="button"
                className="issue-body-link"
                onClick={() => onFocusField?.(issue.field as string)}
              >
                {content}
              </button>
            ) : (
              content
            )}
          </li>
        );
      })}
    </ul>
  );
}