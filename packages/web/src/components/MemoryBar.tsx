import type { MemoryPlan } from "@supertuner/core";
import { formatGib, formatMib } from "../lib/format";

type SegmentKey = "weights" | "adapter" | "optimizer" | "grad" | "activations" | "other";

const SEGMENT_ORDER: { key: SegmentKey; match: RegExp }[] = [
  { key: "weights", match: /weight|base|model|te|text/i },
  { key: "adapter", match: /adapter|lora/i },
  { key: "optimizer", match: /optim/i },
  { key: "grad", match: /grad(ient)?([^a-z]|$)/i },
  { key: "activations", match: /activ|cache|reserv/i },
  { key: "other", match: /.*/ },
];

function segmentKey(label: string): SegmentKey {
  for (const entry of SEGMENT_ORDER) {
    if (entry.match.test(label)) return entry.key;
  }
  return "other";
}

const color = (key: SegmentKey): string => `var(--series-${key})`;

export interface MemoryBarProps {
  memory: MemoryPlan;
  /** Compact mode hides the legend values, for table cells. */
  compact?: boolean;
}

/**
 * VRAM breakdown. Segment width is relative to the larger of the budget and the
 * planned total, so an over-budget plan visibly spills past the budget marker.
 */
export function MemoryBar({ memory, compact }: MemoryBarProps) {
  const lines = memory.lines.filter((line) => Number.isFinite(line.mib) && line.mib > 0);
  const total = lines.reduce((sum, line) => sum + line.mib, 0);
  const budget = memory.vram_budget_mib > 0 ? memory.vram_budget_mib : total;
  const scaleMax = Math.max(budget, total, 1);
  const over = total > budget;

  return (
    <div className="memory">
      <div className="memory-head">
        <span className="eyebrow">VRAM plan</span>
        <span className="mono small">
          <strong>{formatMib(total)}</strong>
          <span className="faint"> / {formatMib(budget)}</span>
        </span>
      </div>

      {lines.length === 0 ? (
        <div className="memory-empty">No memory breakdown reported</div>
      ) : (
        <div
          className="memory-scale"
          role="img"
          aria-label={`VRAM use ${formatMib(total)} of ${formatMib(budget)}${over ? ", over budget" : ""}`}
        >
          {lines.map((line) => {
            const key = segmentKey(line.label);
            return (
              <span
                key={`${line.label}-${line.mib}`}
                className={`memory-seg ${key}`}
                style={{ width: `${(line.mib / scaleMax) * 100}%`, background: color(key) }}
                title={`${line.label}: ${formatMib(line.mib)}${line.note ? ` — ${line.note}` : ""}`}
              />
            );
          })}
          {over ? <span className="memory-over" aria-hidden="true" /> : null}
        </div>
      )}

      {!compact && lines.length > 0 ? (
        <div className="memory-legend">
          {lines.map((line) => {
            const key = segmentKey(line.label);
            return (
              <span className="legend-item" key={`legend-${line.label}-${line.mib}`}>
                <span className="legend-swatch" aria-hidden="true" style={{ background: color(key) }} />
                <span>{line.label}</span>
                <span className="mono faint">{formatGib(line.mib)}</span>
              </span>
            );
          })}
        </div>
      ) : null}

      <div className="memory-foot">
        <span>train peak {formatMib(memory.training_peak_mib)}</span>
        <span>sample peak {formatMib(memory.sampling_peak_mib)}</span>
        <span className={memory.headroom_mib < 0 ? "value" : ""}>
          headroom {formatMib(memory.headroom_mib)}
        </span>
      </div>

      {memory.ram_budget_mib > 0 ? (
        <div className="memory">
          <div className="memory-head">
            <span className="eyebrow">RAM plan</span>
            <span className="mono small">
              <strong>{formatMib(memory.ram_peak_mib)}</strong>
              <span className="faint"> / {formatMib(memory.ram_budget_mib)}</span>
            </span>
          </div>
          <div
            className="memory-scale"
            role="img"
            aria-label={`System RAM use ${formatMib(memory.ram_peak_mib)} of ${formatMib(memory.ram_budget_mib)}`}
          >
            <span
              className="memory-seg ram"
              style={{
                width: `${Math.min(100, (memory.ram_peak_mib / Math.max(memory.ram_budget_mib, 1)) * 100)}%`,
                background: "var(--series-ram)",
              }}
            />
          </div>
          <div className="memory-foot">
            <span>ram peak {formatMib(memory.ram_peak_mib)}</span>
            <span className={memory.ram_headroom_mib < 0 ? "value" : ""}>
              headroom {formatMib(memory.ram_headroom_mib)}
            </span>
          </div>
        </div>
      ) : null}

      {memory.risks.length > 0 ? (
        <ul className="stack-sm small muted">
          {memory.risks.map((risk) => (
            <li key={risk} className="row">
              <span aria-hidden="true">·</span>
              <span>{risk}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}