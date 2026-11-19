import type { Step } from "@supertuner/core";
import { formatDuration, formatMib, formatNumber } from "../lib/format";

export type StepRow = Pick<Step, "step" | "elapsed_s"> & Partial<Omit<Step, "step" | "elapsed_s">>;

export interface StepTableProps {
  steps: readonly StepRow[];
  /** Newest first, capped so a long run does not bloat the DOM. */
  maxRows?: number;
  emptyLabel?: string;
}

export function StepTable({ steps, maxRows = 50, emptyLabel = "No steps yet." }: StepTableProps) {
  if (steps.length === 0) {
    return <p className="small faint">{emptyLabel}</p>;
  }
  const ordered = [...steps].sort((a, b) => b.step - a.step).slice(0, maxRows);

  return (
    <div className="table-wrap">
      <table>
        <caption className="sr-only">Training steps, newest first</caption>
        <thead>
          <tr>
            <th scope="col">Step</th>
            <th scope="col">Loss</th>
            <th scope="col">LR</th>
            <th scope="col">Grad norm</th>
            <th scope="col">Tok/s</th>
            <th scope="col">VRAM peak</th>
            <th scope="col">Elapsed</th>
          </tr>
        </thead>
        <tbody>
          {ordered.map((row) => (
            <tr key={row.step}>
              <th scope="row" className="mono">
                {formatNumber(row.step)}
              </th>
              <td className="mono">{row.loss === null || row.loss === undefined ? "—" : formatNumber(row.loss, 4)}</td>
              <td className="mono">{row.lr === null || row.lr === undefined ? "—" : row.lr.toExponential(2)}</td>
              <td className="mono">
                {row.grad_norm === null || row.grad_norm === undefined ? "—" : formatNumber(row.grad_norm, 3)}
              </td>
              <td className="mono">
                {row.tokens_per_s === null || row.tokens_per_s === undefined ? "—" : formatNumber(row.tokens_per_s, 1)}
              </td>
              <td className="mono">{formatMib(row.vram_peak_mib)}</td>
              <td className="mono">{formatDuration(row.elapsed_s)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}