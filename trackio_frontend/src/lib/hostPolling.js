let rateLimitCooldownUntil = 0;

export const LOCAL_APP_POLL_INTERVAL_MS = 10_000;
export const LOCAL_METRICS_POLL_INTERVAL_MS = 10_000;
export const SPACE_APP_POLL_INTERVAL_MS = 15_000;
export const SPACE_METRICS_POLL_INTERVAL_MS = 15_000;

export function isHfSpaceHost() {
  if (typeof window === "undefined") return false;
  return (window.location.hostname || "")
    .toLowerCase()
    .endsWith(".hf.space");
}

export function registerRateLimitHit() {
  const until = Date.now() + 12000;
  rateLimitCooldownUntil = Math.max(rateLimitCooldownUntil, until);
}

export function isRateLimitCooldownActive() {
  return Date.now() < rateLimitCooldownUntil;
}

export function getAppPollIntervalMs() {
  return isHfSpaceHost()
    ? SPACE_APP_POLL_INTERVAL_MS
    : LOCAL_APP_POLL_INTERVAL_MS;
}

export function getMetricsPollIntervalMs() {
  return isHfSpaceHost()
    ? SPACE_METRICS_POLL_INTERVAL_MS
    : LOCAL_METRICS_POLL_INTERVAL_MS;
}

export function isTabHidden() {
  return typeof document !== "undefined" && document.hidden;
}

/**
 * Serialize a recurring asynchronous observer.
 *
 * setInterval does not wait for an async callback.  Without this guard, a slow
 * SQLite read, artifact request, or browser render permits the next interval
 * to start and creates an unbounded queue of equivalent work.  A skipped tick
 * is correct for telemetry: the following completed refresh observes the
 * latest durable state.
 */
export function createSingleFlightPoller() {
  let running = false;
  return async function run(task) {
    if (running) return false;
    running = true;
    try {
      await task();
      return true;
    } finally {
      running = false;
    }
  };
}
