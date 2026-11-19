import { useMemo, useState } from "react";
import { formatNumber } from "../lib/format";

/**
 * Dependency-free loss chart. `buildLossChart` is pure so the path maths can be
 * tested without a DOM: it handles zero points, a single point, gaps caused by
 * null losses, and flat loss values.
 */

export interface LossPoint {
  step: number;
  loss: number | null;
}

export interface ChartPoint {
  step: number;
  loss: number;
  x: number;
  y: number;
}

export interface AxisTick {
  value: number;
  position: number;
  label: string;
}

export interface LossChartGeometry {
  width: number;
  height: number;
  plot: { left: number; top: number; right: number; bottom: number };
  path: string;
  area: string;
  points: ChartPoint[];
  segments: number;
  xTicks: AxisTick[];
  yTicks: AxisTick[];
  domain: { minStep: number; maxStep: number; minLoss: number; maxLoss: number } | null;
}

export interface BuildOptions {
  width?: number;
  height?: number;
  padding?: { top: number; right: number; bottom: number; left: number };
  yTickCount?: number;
  xTickCount?: number;
}

const DEFAULT_PADDING = { top: 12, right: 16, bottom: 26, left: 46 };

function tickLabel(value: number, digits: number): string {
  const abs = Math.abs(value);
  if (abs >= 10000) return `${formatNumber(value / 1000, 1)}k`;
  return formatNumber(value, digits);
}

export function buildLossChart(points: readonly LossPoint[], options: BuildOptions = {}): LossChartGeometry {
  const width = options.width ?? 720;
  const height = options.height ?? 240;
  const padding = options.padding ?? DEFAULT_PADDING;
  const yTickCount = options.yTickCount ?? 4;
  const xTickCount = options.xTickCount ?? 5;
  const plot = { left: padding.left, top: padding.top, right: width - padding.right, bottom: height - padding.bottom };
  const plotWidth = Math.max(1, plot.right - plot.left);
  const plotHeight = Math.max(1, plot.bottom - plot.top);

  const empty: LossChartGeometry = {
    width,
    height,
    plot,
    path: "",
    area: "",
    points: [],
    segments: 0,
    xTicks: [],
    yTicks: [],
    domain: null,
  };

  // Deduplicate by step (last value wins). A null loss is kept as a gap marker.
  const byStep = new Map<number, number | null>();
  for (const point of points) {
    if (!Number.isFinite(point.step)) continue;
    const loss = point.loss !== null && Number.isFinite(point.loss) ? point.loss : null;
    byStep.set(point.step, loss);
  }
  const series = [...byStep.entries()]
    .map(([step, loss]) => ({ step, loss }))
    .sort((a, b) => a.step - b.step);
  const ordered = series.filter((entry): entry is { step: number; loss: number } => entry.loss !== null);
  if (ordered.length === 0) return empty;

  let minStep = ordered[0]?.step ?? 0;
  let maxStep = ordered[ordered.length - 1]?.step ?? 0;
  if (minStep === maxStep) {
    minStep -= 1;
    maxStep += 1;
  }

  const losses = ordered.map((point) => point.loss);
  let minLoss = Math.min(...losses);
  let maxLoss = Math.max(...losses);
  if (minLoss === maxLoss) {
    const pad = Math.max(Math.abs(minLoss) * 0.1, 0.05);
    minLoss -= pad;
    maxLoss += pad;
  } else {
    const pad = (maxLoss - minLoss) * 0.08;
    minLoss -= pad;
    maxLoss += pad;
  }

  const xFor = (step: number): number => plot.left + ((step - minStep) / (maxStep - minStep)) * plotWidth;
  const yFor = (loss: number): number => plot.bottom - ((loss - minLoss) / (maxLoss - minLoss)) * plotHeight;

  const chartPoints: ChartPoint[] = ordered.map((point) => ({
    step: point.step,
    loss: point.loss,
    x: xFor(point.step),
    y: yFor(point.loss),
  }));

  // A null in the series starts a new subpath, so gaps read as gaps rather than
  // as a straight line through missing data.
  let path = "";
  let area = "";
  let segments = 0;
  let open = false;
  let lastX = plot.left;
  for (const entry of series) {
    if (entry.loss === null) {
      if (open) {
        area += ` L ${round(lastX)} ${round(plot.bottom)} Z`;
        open = false;
      }
      continue;
    }
    const x = xFor(entry.step);
    const y = yFor(entry.loss);
    if (!open) {
      path += `${path ? " " : ""}M ${round(x)} ${round(y)}`;
      area += `${area ? " " : ""}M ${round(x)} ${round(plot.bottom)} L ${round(x)} ${round(y)}`;
      segments += 1;
      open = true;
    } else {
      path += ` L ${round(x)} ${round(y)}`;
      area += ` L ${round(x)} ${round(y)}`;
    }
    lastX = x;
  }
  if (open) area += ` L ${round(lastX)} ${round(plot.bottom)} Z`;

  const yTicks: AxisTick[] = [];
  for (let index = 0; index <= yTickCount; index += 1) {
    const ratio = index / yTickCount;
    const value = minLoss + (maxLoss - minLoss) * ratio;
    yTicks.push({ value, position: plot.bottom - ratio * plotHeight, label: tickLabel(value, 3) });
  }

  const xTicks: AxisTick[] = [];
  const xCount = Math.max(1, Math.min(xTickCount, 8));
  for (let index = 0; index < xCount; index += 1) {
    const ratio = index / (xCount - 1 || 1);
    const rawStep = minStep + (maxStep - minStep) * ratio;
    const step = Math.round(rawStep);
    xTicks.push({ value: step, position: xFor(step), label: formatNumber(step) });
  }

  return {
    width,
    height,
    plot,
    path: path.trim(),
    area: area.trim(),
    points: chartPoints,
    segments,
    xTicks,
    yTicks,
    domain: { minStep, maxStep, minLoss, maxLoss },
  };
}

function round(value: number): number {
  return Math.round(value * 100) / 100;
}

export interface LossChartProps {
  points: readonly LossPoint[];
  height?: number;
  /** Extra readout shown in the tooltip, e.g. learning rate. */
  ariaLabel?: string;
  idleLabel?: string;
}

export function LossChart({ points, height, ariaLabel, idleLabel = "No loss data yet" }: LossChartProps) {
  const geometry = useMemo(() => buildLossChart(points, height ? { height } : {}), [points, height]);
  const [cursor, setCursor] = useState<number | null>(null);

  const active = cursor === null ? null : geometry.points[cursor] ?? null;
  const hasData = geometry.points.length > 0;

  const moveTo = (index: number | null): void => {
    if (!hasData) return;
    if (index === null) {
      setCursor(null);
      return;
    }
    const clamped = Math.max(0, Math.min(geometry.points.length - 1, index));
    setCursor(clamped);
  };

  if (!hasData) {
    return (
      <div className="chart">
        <div className="chart-empty">
          <span>{idleLabel}</span>
          <span className="xs faint">Points appear as training steps report in.</span>
        </div>
      </div>
    );
  }

  const summary =
    ariaLabel ??
    `Loss over ${geometry.points.length} steps, from ${formatNumber(geometry.points[0]?.loss ?? 0, 3)} to ${formatNumber(
      geometry.points[geometry.points.length - 1]?.loss ?? 0,
      3,
    )}`;

  return (
    <div className="chart">
      <div className="chart-plot">
        <svg
          viewBox={`0 0 ${geometry.width} ${geometry.height}`}
          className="chart-svg"
          role="img"
          aria-label={summary}
          tabIndex={0}
          onFocus={() => moveTo(geometry.points.length - 1)}
          onBlur={() => setCursor(null)}
          onKeyDown={(event) => {
            if (event.key === "ArrowLeft") {
              event.preventDefault();
              moveTo((cursor ?? geometry.points.length) - 1);
            } else if (event.key === "ArrowRight") {
              event.preventDefault();
              moveTo((cursor ?? -1) + 1);
            } else if (event.key === "Escape") {
              setCursor(null);
            }
          }}
          onPointerMove={(event) => {
            const rect = event.currentTarget.getBoundingClientRect();
            if (rect.width === 0) return;
            const x = ((event.clientX - rect.left) / rect.width) * geometry.width;
            let bestIndex = 0;
            let bestDistance = Number.POSITIVE_INFINITY;
            geometry.points.forEach((point, index) => {
              const distance = Math.abs(point.x - x);
              if (distance < bestDistance) {
                bestDistance = distance;
                bestIndex = index;
              }
            });
            moveTo(bestIndex);
          }}
          onPointerLeave={() => setCursor(null)}
        >
          <g className="chart-grid">
            {geometry.yTicks.map((tick, index) => (
              <line key={`yg-${index}`} x1={geometry.plot.left} x2={geometry.plot.right} y1={tick.position} y2={tick.position} />
            ))}
          </g>

          <g className="chart-axis">
            {geometry.yTicks.map((tick, index) => (
              <text key={`yt-${index}`} x={geometry.plot.left - 8} y={tick.position + 3} textAnchor="end">
                {tick.label}
              </text>
            ))}
            {geometry.xTicks.map((tick, index) => (
              <text key={`xt-${index}`} x={tick.position} y={geometry.plot.bottom + 16} textAnchor="middle">
                {tick.label}
              </text>
            ))}
          </g>

          <line
            x1={geometry.plot.left}
            x2={geometry.plot.right}
            y1={geometry.plot.bottom}
            y2={geometry.plot.bottom}
            stroke="var(--chart-axis)"
            strokeWidth={1}
          />

          {geometry.area ? <path d={geometry.area} className="chart-area" fill="var(--chart-line-soft)" stroke="none" /> : null}
          <path d={geometry.path} className="chart-path" />

          {active ? (
            <>
              <line
                className="chart-hover-line"
                x1={active.x}
                x2={active.x}
                y1={geometry.plot.top}
                y2={geometry.plot.bottom}
              />
              <circle className="chart-dot" cx={active.x} cy={active.y} r={3.5} />
            </>
          ) : geometry.points.length <= 40 ? (
            geometry.points.map((point) => (
              <circle key={`dot-${point.step}`} className="chart-dot" cx={point.x} cy={point.y} r={geometry.points.length === 1 ? 4 : 2.4} />
            ))
          ) : null}
        </svg>

        {active ? (
          <div
            className="chart-tooltip"
            style={{
              left: `${(active.x / geometry.width) * 100}%`,
              top: `${(active.y / geometry.height) * 100}%`,
            }}
          >
            <div className="row spread">
              <span className="faint">step</span>
              <span className="mono">{formatNumber(active.step)}</span>
            </div>
            <div className="row spread">
              <span className="faint">loss</span>
              <span className="mono">{formatNumber(active.loss, 4)}</span>
            </div>
          </div>
        ) : null}
      </div>

      <p className="sr-only" aria-live="polite">
        {active ? `Step ${formatNumber(active.step)}, loss ${formatNumber(active.loss, 3)}` : ""}
      </p>

      <div className="memory-foot">
        <span>step {formatNumber(geometry.domain?.minStep ?? 0)}–{formatNumber(geometry.domain?.maxStep ?? 0)}</span>
        <span>{geometry.segments} segment{geometry.segments === 1 ? "" : "s"}</span>
      </div>
    </div>
  );
}