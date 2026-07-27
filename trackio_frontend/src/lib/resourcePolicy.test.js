import { describe, expect, test } from "vitest";
import {
  DEFAULT_OPEN_METRIC_GROUPS,
  DEFAULT_SMOOTHING,
  MAX_METRIC_POINTS_PER_RUN,
  isDefaultMetricGroupOpen,
} from "./resourcePolicy.js";

describe("dashboard resource authority", () => {
  test("bounds each selected run before browser materialization", () => {
    expect(MAX_METRIC_POINTS_PER_RUN).toBeGreaterThanOrEqual(500);
    expect(MAX_METRIC_POINTS_PER_RUN).toBeLessThanOrEqual(1_000);
  });

  test("does not duplicate every observation with smoothing by default", () => {
    expect(DEFAULT_SMOOTHING).toBe(0);
  });

  test("lazily opens only the compact core training groups", () => {
    expect(DEFAULT_OPEN_METRIC_GROUPS).toEqual(["train", "progress"]);
    expect(isDefaultMetricGroupOpen("train")).toBe(true);
    expect(isDefaultMetricGroupOpen("progress")).toBe(true);
    expect(isDefaultMetricGroupOpen("architecture")).toBe(false);
    expect(isDefaultMetricGroupOpen("pc_rasl")).toBe(false);
  });
});
