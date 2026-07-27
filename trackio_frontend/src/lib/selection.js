export function latestOnlySelection(filteredRunIds) {
  if (!filteredRunIds || filteredRunIds.length === 0) return [];
  return [filteredRunIds[0]];
}

export function pruneRunCache(cache, selectedRuns) {
  const active = new Set(
    (selectedRuns ?? []).map((run) =>
      typeof run === "string" ? run : (run?.id ?? run?.name),
    ),
  );
  for (const key of cache.keys()) {
    if (!active.has(key)) cache.delete(key);
  }
  return cache;
}

export function reconcileSelectedRuns(prevSelected, newOrderedIds, prevOrderedIds) {
  const prev = prevSelected ?? [];
  const ordered = newOrderedIds ?? [];
  const prevOrdered = prevOrderedIds ?? [];
  const newIdSet = new Set(ordered);
  const kept = prev.filter((r) => newIdSet.has(r));

  if (prev.length === 0 || kept.length === 0) {
    // A dashboard is an observer, not a sweep renderer.  Selecting every
    // historical run on first load makes both the API and Vega instantiate
    // work proportional to the lifetime of the project.  The newest run is
    // the useful and bounded default; comparison remains an explicit choice.
    return latestOnlySelection(ordered);
  }

  const allPrevSelected =
    prevOrdered.length > 0 && prev.length === prevOrdered.length;
  if (allPrevSelected) {
    const keptSet = new Set(kept);
    const additions = ordered.filter((r) => !keptSet.has(r));
    return [...kept, ...additions];
  }

  return kept;
}
