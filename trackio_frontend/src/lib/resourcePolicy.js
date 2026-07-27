/**
 * Resource authority for the live dashboard.
 *
 * These bounds affect only observer fidelity.  Trackio's SQLite database and
 * append-only training mirrors retain the complete metric history.
 */
export const MAX_METRIC_POINTS_PER_RUN = 1_000;
export const DEFAULT_SMOOTHING = 0;
export const DEFAULT_OPEN_METRIC_GROUPS = Object.freeze([
  "train",
  "progress",
]);

export function isDefaultMetricGroupOpen(groupName) {
  return DEFAULT_OPEN_METRIC_GROUPS.includes(groupName);
}
