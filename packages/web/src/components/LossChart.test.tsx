import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LossChart, buildLossChart, type LossPoint } from "./LossChart";

function series(count: number): LossPoint[] {
  return Array.from({ length: count }, (_, index) => ({ step: index + 1, loss: 3 - index * 0.01 }));
}

describe("buildLossChart", () => {
  it("returns an empty geometry for no points", () => {
    const chart = buildLossChart([]);
    expect(chart.path).toBe("");
    expect(chart.area).toBe("");
    expect(chart.points).toEqual([]);
    expect(chart.segments).toBe(0);
    expect(chart.domain).toBeNull();
    expect(chart.xTicks).toEqual([]);
    expect(chart.yTicks).toEqual([]);
  });

  it("treats an all-null series as empty", () => {
    const chart = buildLossChart([
      { step: 1, loss: null },
      { step: 2, loss: null },
    ]);
    expect(chart.path).toBe("");
    expect(chart.domain).toBeNull();
  });

  it("draws a single point as one point and a padded domain", () => {
    const chart = buildLossChart([{ step: 10, loss: 2.5 }]);
    expect(chart.points).toHaveLength(1);
    expect(chart.segments).toBe(1);
    expect(chart.path.startsWith("M ")).toBe(true);
    expect(chart.path).not.toContain(" L ");
    const [point] = chart.points;
    expect(point?.step).toBe(10);
    expect(chart.domain).not.toBeNull();
    expect(chart.domain!.minStep).toBeLessThan(10);
    expect(chart.domain!.maxStep).toBeGreaterThan(10);
    expect(chart.domain!.minLoss).toBeLessThan(2.5);
    expect(chart.domain!.maxLoss).toBeGreaterThan(2.5);
    expect(chart.yTicks).toHaveLength(5);
  });

  it("builds a continuous path for many points", () => {
    const chart = buildLossChart(series(50));
    expect(chart.points).toHaveLength(50);
    expect(chart.segments).toBe(1);
    expect(chart.path.match(/M /g)).toHaveLength(1);
    expect(chart.path.match(/ L /g)).toHaveLength(49);
    expect(chart.area.endsWith("Z")).toBe(true);
    expect(chart.xTicks).toHaveLength(5);
    for (let index = 1; index < chart.xTicks.length; index += 1) {
      expect(chart.xTicks[index]!.position).toBeGreaterThan(chart.xTicks[index - 1]!.position);
    }
  });

  it("breaks the path at null losses and still fills the area", () => {
    const chart = buildLossChart([
      { step: 1, loss: 3 },
      { step: 2, loss: null },
      { step: 3, loss: 2.6 },
      { step: 4, loss: null },
      { step: 5, loss: 2.4 },
    ]);
    expect(chart.points).toHaveLength(3);
    expect(chart.segments).toBe(3);
    expect(chart.path.match(/M /g)).toHaveLength(3);
    expect(chart.path.match(/ L /g)).toBeNull();
    expect(chart.area.match(/Z/g)).toHaveLength(3);
  });

  it("keeps the last value when a step repeats", () => {
    const chart = buildLossChart([
      { step: 1, loss: 3 },
      { step: 1, loss: 2.2 },
      { step: 2, loss: 2 },
    ]);
    expect(chart.points).toHaveLength(2);
    expect(chart.points[0]?.loss).toBe(2.2);
  });

  it("ignores non-finite losses", () => {
    const chart = buildLossChart([
      { step: 1, loss: 3 },
      { step: 2, loss: Number.NaN },
      { step: 3, loss: Number.POSITIVE_INFINITY },
    ]);
    expect(chart.points).toHaveLength(1);
    expect(chart.segments).toBe(1);
  });

  it("pads a flat loss series so the line is visible", () => {
    const chart = buildLossChart([
      { step: 1, loss: 1 },
      { step: 2, loss: 1 },
      { step: 3, loss: 1 },
    ]);
    expect(chart.domain!.maxLoss).toBeGreaterThan(1);
    expect(chart.domain!.minLoss).toBeLessThan(1);
  });

  it("respects explicit height and tick counts", () => {
    const chart = buildLossChart(series(10), { height: 320, yTickCount: 3 });
    expect(chart.height).toBe(320);
    expect(chart.yTicks).toHaveLength(4);
    expect(chart.plot.bottom).toBeLessThan(320);
  });
});

describe("LossChart component", () => {
  it("shows the idle label when there is nothing to draw", () => {
    render(<LossChart points={[]} />);
    expect(screen.getByText("No loss data yet")).toBeInTheDocument();
  });

  it("renders an accessible svg once points exist", () => {
    render(<LossChart points={series(8)} ariaLabel="Loss for smoke run" />);
    const svg = screen.getByRole("img", { name: "Loss for smoke run" });
    expect(svg).toBeInTheDocument();
    expect(svg).toHaveAttribute("tabindex", "0");
    expect(svg.querySelector(".chart-path")).not.toBeNull();
    expect(screen.queryByText("No loss data yet")).not.toBeInTheDocument();
  });

  it("renders a single point without a line segment", () => {
    render(<LossChart points={[{ step: 5, loss: 1.25 }]} />);
    const svg = screen.getByRole("img");
    expect(svg.querySelector(".chart-path")?.getAttribute("d") ?? "").toContain("M ");
    expect(svg.querySelectorAll(".chart-dot")).toHaveLength(1);
  });
});