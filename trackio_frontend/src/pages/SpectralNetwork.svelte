<script>
  import { onMount } from "svelte";
  import {
    getArtifactBlobUrl,
    getArtifactManifest,
    getRunArtifacts,
  } from "../lib/api.js";
  import {
    getAppPollIntervalMs,
    isRateLimitCooldownActive,
    isTabHidden,
  } from "../lib/hostPolling.js";

  let {
    project = null,
    selectedRuns = [],
    allRuns = [],
    realtimeEnabled = true,
  } = $props();

  let frame = $state(null);
  let evidence = $state(null);
  let loading = $state(false);
  let error = $state("");
  let loadedKey = $state("");
  let requestId = 0;

  function runKey(run) {
    return run?.id ?? run?.name ?? "";
  }

  let selectedRun = $derived(selectedRuns[0] ?? allRuns[0] ?? null);
  let viewerUrl = $derived(`${window.__trackio_base || ""}/mrrn-spectral-view.html`);

  function sendEvidence() {
    if (!frame?.contentWindow || !evidence) return;
    frame.contentWindow.postMessage(
      { type: "mrcra-cognitive-spectral-evidence", evidenceJson: JSON.stringify(evidence) },
      window.location.origin,
    );
  }

  async function loadEvidence({ quiet = false } = {}) {
    const id = ++requestId;
    const run = selectedRun;
    if (!project || !run) {
      evidence = null;
      error = "Select a Trackio run to inspect its spectral state.";
      return;
    }
    if (!quiet) loading = true;
    try {
      const linked = await getRunArtifacts(project, run);
      if (id !== requestId) return;
      const candidates = (linked?.output || [])
        .filter((item) => item.name === "mrcra-cognitive-spectral-evidence")
        .sort((a, b) => b.version - a.version);
      if (!candidates.length) {
        evidence = null;
        loadedKey = "";
        error = "This run has not published a spectral snapshot yet. The trainer publishes one after its first optimizer step.";
        return;
      }
      const latest = candidates[0];
      const key = `${runKey(run)}:${latest.version}`;
      if (quiet && key === loadedKey) return;
      const record = await getArtifactManifest(project, latest.name, `v${latest.version}`);
      if (id !== requestId) return;
      const file = (record?.manifest || []).find((item) =>
        item.path.endsWith("mrcra-cognitive-spectral-data.json"),
      );
      if (!file) throw new Error("Spectral artifact contains no evidence JSON.");
      const response = await fetch(getArtifactBlobUrl(project, file.digest), {
        credentials: "include",
        cache: "no-store",
      });
      if (!response.ok) throw new Error(`Evidence download failed (${response.status}).`);
      evidence = await response.json();
      loadedKey = key;
      error = "";
      sendEvidence();
    } catch (cause) {
      if (id === requestId) error = cause?.message || "Could not load spectral evidence.";
    } finally {
      if (id === requestId) loading = false;
    }
  }

  $effect(() => {
    project;
    runKey(selectedRun);
    loadedKey = "";
    loadEvidence();
  });

  $effect(() => {
    evidence;
    sendEvidence();
  });

  onMount(() => {
    const timer = setInterval(() => {
      if (!realtimeEnabled || isTabHidden() || isRateLimitCooldownActive()) return;
      loadEvidence({ quiet: true });
    }, getAppPollIntervalMs());
    return () => clearInterval(timer);
  });
</script>

<div class="spectral-page">
  {#if loading && !evidence}
    <div class="status">Loading the latest spectral snapshot…</div>
  {:else if error && !evidence}
    <div class="status error">{error}</div>
  {/if}
  {#if evidence}
    <div class="snapshot-line">
      <span>{selectedRun?.name || "Selected run"}</span>
      <span>snapshot step {evidence.checkpoint?.step ?? "—"}</span>
      <span>{(evidence.checkpoint?.tokens_seen ?? 0).toLocaleString()} tokens</span>
      {#if error}<span class="refresh-error">Refresh: {error}</span>{/if}
    </div>
  {/if}
  <iframe
    bind:this={frame}
    src={viewerUrl}
    title="MRCRA cognitive-spectral network instruments"
    onload={sendEvidence}
  ></iframe>
</div>

<style>
  .spectral-page {
    width: 100%;
    min-width: 0;
    display: flex;
    flex-direction: column;
    background: var(--background-fill-primary, #fff);
  }
  .snapshot-line {
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
    padding: 9px 18px;
    border-bottom: 1px solid var(--border-color-primary, #e5e7eb);
    color: var(--body-text-color-subdued, #6b7280);
    font-size: var(--text-sm, 12px);
  }
  .snapshot-line span:first-child {
    color: var(--body-text-color, #1f2937);
    font-weight: 500;
  }
  .refresh-error, .error { color: var(--error-text-color, #b91c1c); }
  .status { padding: 24px; color: var(--body-text-color-subdued, #6b7280); }
  iframe { flex: 1; width: 100%; border: 0; background: transparent; }
</style>
