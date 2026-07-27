<script>
  import { onMount } from "svelte";
  import { getArtifactBlobUrl, getArtifactManifest, getRunArtifacts } from "../lib/api.js";
  import {
    createSingleFlightPoller,
    getAppPollIntervalMs,
    isRateLimitCooldownActive,
    isTabHidden,
  } from "../lib/hostPolling.js";
  import {
    causalTimeline,
    deliberationLattice,
    invariantRows,
    reconstructionRows,
    viabilityRows,
  } from "../lib/cognitiveViews.js";

  let { project = null, selectedRuns = [], allRuns = [], realtimeEnabled = true } = $props();
  let evidence = $state(null);
  let loading = $state(false);
  let error = $state("");
  let loadedKey = $state("");
  let view = $state("reconstruction");
  let requestId = 0;
  const runEvidencePoll = createSingleFlightPoller();
  const views = [
    ["reconstruction", "Reconstructive Descent"],
    ["deliberation", "Deliberation Lattice"],
    ["viability", "Viability Envelope"],
    ["invariants", "Invariant Transfer"],
    ["timeline", "Cognitive Causal Timeline"],
  ];

  let selectedRun = $derived(selectedRuns[0] ?? allRuns[0] ?? null);
  let cognitive = $derived(evidence?.cognitive || {});
  let reconstructions = $derived(reconstructionRows(cognitive));
  let lattice = $derived(deliberationLattice(cognitive));
  let viability = $derived(viabilityRows(cognitive));
  let invariants = $derived(invariantRows(cognitive));
  let timeline = $derived(causalTimeline(cognitive));

  function runKey(run) { return run?.id ?? run?.name ?? ""; }
  function percent(value) { return `${Math.max(0, Math.min(100, Number(value || 0) * 100)).toFixed(0)}%`; }

  async function loadEvidence({ quiet = false } = {}) {
    const id = ++requestId;
    const run = selectedRun;
    if (!project || !run) {
      evidence = null;
      error = "Select a Trackio run to inspect its cognitive state.";
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
        error = "This run has not published an MRCRA cognitive snapshot yet.";
        return;
      }
      const latest = candidates[0];
      const key = `${runKey(run)}:${latest.version}`;
      if (quiet && key === loadedKey) return;
      const record = await getArtifactManifest(project, latest.name, `v${latest.version}`);
      if (id !== requestId) return;
      const file = (record?.manifest || []).find((item) => item.path.endsWith("mrcra-cognitive-spectral-data.json"));
      if (!file) throw new Error("MRCRA artifact contains no evidence JSON.");
      const response = await fetch(getArtifactBlobUrl(project, file.digest), {
        credentials: "include", cache: "no-store",
      });
      if (!response.ok) throw new Error(`Evidence download failed (${response.status}).`);
      const next = await response.json();
      if (!next?.cognitive) throw new Error("Snapshot predates cognitive evidence schema 3.");
      evidence = next;
      loadedKey = key;
      error = "";
    } catch (cause) {
      if (id === requestId) error = cause?.message || "Could not load cognitive evidence.";
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

  onMount(() => {
    const timer = setInterval(() => {
      if (!realtimeEnabled || isTabHidden() || isRateLimitCooldownActive()) return;
      runEvidencePoll(() => loadEvidence({ quiet: true })).catch((cause) => {
        console.error("Failed to poll cognitive evidence:", cause);
      });
    }, getAppPollIntervalMs());
    return () => clearInterval(timer);
  });
</script>

<div class="cognitive-page">
  {#if loading && !evidence}<div class="status">Loading cognitive evidence…</div>{/if}
  {#if error && !evidence}<div class="status error">{error}</div>{/if}
  {#if evidence}
    <header>
      <div><strong>{selectedRun?.name || "Selected run"}</strong> · step {evidence.checkpoint?.step ?? "—"}</div>
      <div class="authority">Diagnostic projection of immutable receipts · never a control input</div>
    </header>
    <nav class="view-tabs" aria-label="MRCRA cognitive views">
      {#each views as item}
        <button class:active={view === item[0]} onclick={() => (view = item[0])}>{item[1]}</button>
      {/each}
    </nav>
    <main>
      {#if view === "reconstruction"}
        <h2>Abstraction → traces and evidence → localized reconstructed graph</h2>
        {#if reconstructions.length}
          <div class="card-grid">
            {#each reconstructions as item}
              <article class="card">
                <h3>Abstraction {item.abstraction} · reconstruction {item.slot}</h3>
                <div class="flow"><span>abstraction</span><b>→</b><span>evidence conditioned</span><b>→</b><span class="derived">reconstructed</span></div>
                <meter min="0" max="1" value={item.historical_fidelity}></meter><label>historical fidelity {percent(item.historical_fidelity)}</label>
                <meter min="0" max="1" value={item.structural_plausibility}></meter><label>structural plausibility {percent(item.structural_plausibility)}</label>
                <meter min="0" max="1" value={item.evidence_agreement}></meter><label>evidence agreement {percent(item.evidence_agreement)}</label>
                <p>scale {item.scale} · depth {item.depth} · provenance {item.provenance}</p>
              </article>
            {/each}
          </div>
        {:else}<div class="empty">No reconstruction receipt in this snapshot.</div>{/if}
      {:else if view === "deliberation"}
        <h2>Routed hypotheses × candidate actions</h2>
        {#if lattice.length}
          <table><thead><tr><th>Hypothesis</th><th>Posterior</th><th>Action</th><th>Utility</th><th>Information</th><th>Tail risk</th><th>Gate</th></tr></thead>
            <tbody>{#each lattice as item}<tr class:selected={item.selected}><td>{item.hypothesis}{item.unknown ? " (unknown)" : ""}</td><td>{percent(item.posterior)}</td><td>{item.action}</td><td>{item.utility.toFixed(3)}</td><td>{item.information_gain.toFixed(3)}</td><td>{item.tail_risk.toFixed(3)}</td><td class:authorized={item.authorized}>{item.authorized ? "authorized" : "blocked"}</td></tr>{/each}</tbody>
          </table>
        {:else}<div class="empty">No candidate lattice in this snapshot.</div>{/if}
      {:else if view === "viability"}
        <h2>Measured regulated variables and hard violations</h2>
        {#if viability.length}
          <div class="card-grid">{#each viability as item}<article class:violation={item.hard_violation} class="card"><h3>Channel {item.channel}</h3><div class="envelope"><i style={`left:${percent(item.target_low)};width:${percent(item.target_high - item.target_low)}`}></i><b style={`left:${percent(item.value)}`}></b></div><p>value {item.value.toFixed(3)} · target {item.target_low.toFixed(3)}–{item.target_high.toFixed(3)}</p><strong>{item.hard_violation ? "HARD VIOLATION" : "inside hard envelope"}</strong></article>{/each}</div>
        {:else}<div class="empty">No application-authoritative viability channels are active.</div>{/if}
      {:else if view === "invariants"}
        <h2>Role-normalized invariant candidates and transfer evidence</h2>
        {#if invariants.length}
          <table><thead><tr><th>Slot</th><th>Status</th><th>Code gain</th><th>Distortion</th><th>Transfer utility</th><th>Counterexamples</th></tr></thead><tbody>{#each invariants as item}<tr><td>{item.slot}</td><td>{item.status}</td><td>{item.code_gain_bits.toFixed(2)} bits</td><td>{item.distortion.toFixed(4)}</td><td>{item.transfer_utility.toFixed(4)}</td><td>{item.counterexample_search_completed ? "searched" : "pending"}</td></tr>{/each}</tbody></table>
        {:else}<div class="empty">No invariant proposal in this snapshot.</div>{/if}
      {:else}
        <h2>Observation → internal operation → authorization → feedback</h2>
        {#if timeline.length}<ol class="timeline">{#each timeline as item}<li class:failed={!item.success}><span>{item.stage}</span><strong>{item.label}</strong><small>{item.detail}</small></li>{/each}</ol>{:else}<div class="empty">No causal receipts in this snapshot.</div>{/if}
      {/if}
    </main>
  {/if}
</div>

<style>
  .cognitive-page { width:100%; min-width:0; overflow:auto; background:var(--background-fill-primary,#fff); color:var(--body-text-color,#1f2937); }
  header { display:flex; justify-content:space-between; gap:20px; padding:12px 20px; border-bottom:1px solid var(--border-color-primary,#e5e7eb); }
  .authority { color:var(--body-text-color-subdued,#6b7280); font-size:12px; }
  .view-tabs { display:flex; gap:4px; flex-wrap:wrap; padding:10px 18px; border-bottom:1px solid var(--border-color-primary,#e5e7eb); }
  button { border:0; border-radius:6px; padding:8px 11px; background:transparent; color:inherit; cursor:pointer; }
  button:hover, button.active { background:var(--button-secondary-background-fill-hover,#e5e7eb); }
  button.active { font-weight:650; box-shadow:inset 0 -2px var(--color-accent,#7c3aed); }
  main { padding:18px 22px 40px; } h2 { font-size:17px; margin:0 0 16px; }
  .card-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(270px,1fr)); gap:14px; }
  .card { border:1px solid var(--border-color-primary,#ddd); border-radius:10px; padding:14px; background:var(--background-fill-secondary,#f8fafc); }
  .card h3 { margin:0 0 12px; font-size:14px; } .card p { color:var(--body-text-color-subdued,#6b7280); font-size:12px; }
  .flow { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:14px; } .flow span { padding:5px 8px; border-radius:12px; background:#dbeafe; color:#1e40af; } .flow .derived { background:#ede9fe; color:#5b21b6; }
  meter { width:100%; height:10px; display:block; margin-top:8px; } label { font-size:11px; color:var(--body-text-color-subdued,#6b7280); }
  table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; } th,td { text-align:left; padding:9px 10px; border-bottom:1px solid var(--border-color-primary,#e5e7eb); font-size:12px; } th { color:var(--body-text-color-subdued,#6b7280); }
  tr.selected { background:color-mix(in srgb,var(--color-accent,#7c3aed) 12%,transparent); } td.authorized { color:#15803d; font-weight:650; }
  .envelope { position:relative; height:16px; border-radius:8px; background:#fee2e2; overflow:hidden; margin:20px 0 10px; } .envelope i { position:absolute; height:100%; background:#bbf7d0; } .envelope b { position:absolute; top:-3px; width:3px; height:22px; background:#111827; }
  .violation { border-color:#dc2626; } .timeline { display:flex; gap:8px; overflow:auto; padding:10px 0; list-style:none; } .timeline li { min-width:150px; border:1px solid #86efac; border-radius:8px; padding:10px; position:relative; } .timeline li:not(:last-child)::after { content:"→"; position:absolute; right:-9px; top:26px; } .timeline li.failed { border-color:#fca5a5; } .timeline span,.timeline small { display:block; color:var(--body-text-color-subdued,#6b7280); font-size:10px; text-transform:uppercase; } .timeline strong { display:block; margin:5px 0; font-size:13px; }
  .status,.empty { padding:24px; color:var(--body-text-color-subdued,#6b7280); } .error { color:var(--error-text-color,#b91c1c); }
</style>
