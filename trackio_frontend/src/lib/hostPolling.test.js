import { describe, expect, test, vi } from "vitest";
import {
  LOCAL_APP_POLL_INTERVAL_MS,
  LOCAL_METRICS_POLL_INTERVAL_MS,
  SPACE_APP_POLL_INTERVAL_MS,
  SPACE_METRICS_POLL_INTERVAL_MS,
  createSingleFlightPoller,
  getAppPollIntervalMs,
  getMetricsPollIntervalMs,
} from "./hostPolling.js";

describe("bounded dashboard polling", () => {
  test("uses low-frequency local polling instead of one-second full refreshes", () => {
    expect(getAppPollIntervalMs()).toBe(LOCAL_APP_POLL_INTERVAL_MS);
    expect(getMetricsPollIntervalMs()).toBe(LOCAL_METRICS_POLL_INTERVAL_MS);
    expect(LOCAL_APP_POLL_INTERVAL_MS).toBeGreaterThanOrEqual(10_000);
    expect(LOCAL_METRICS_POLL_INTERVAL_MS).toBeGreaterThanOrEqual(10_000);
    expect(SPACE_APP_POLL_INTERVAL_MS).toBeGreaterThanOrEqual(10_000);
    expect(SPACE_METRICS_POLL_INTERVAL_MS).toBeGreaterThanOrEqual(10_000);
  });

  test("allows at most one in-flight refresh and recovers after completion", async () => {
    const poll = createSingleFlightPoller();
    let release;
    const firstTask = vi.fn(
      () => new Promise((resolve) => {
        release = resolve;
      }),
    );
    const overlappingTask = vi.fn(async () => {});

    const first = poll(firstTask);
    await Promise.resolve();
    expect(await poll(overlappingTask)).toBe(false);
    expect(firstTask).toHaveBeenCalledTimes(1);
    expect(overlappingTask).not.toHaveBeenCalled();

    release();
    expect(await first).toBe(true);
    expect(await poll(overlappingTask)).toBe(true);
    expect(overlappingTask).toHaveBeenCalledTimes(1);
  });

  test("releases the single-flight gate after an error", async () => {
    const poll = createSingleFlightPoller();
    await expect(
      poll(async () => {
        throw new Error("transient");
      }),
    ).rejects.toThrow("transient");
    expect(await poll(async () => {})).toBe(true);
  });
});
