#!/usr/bin/env python3
"""Build the deterministic MRCRA source-to-code evidence ledger."""

from __future__ import annotations

import json
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrrn.traceability import inventory  # noqa: E402


SPEC = ROOT / "outputs" / "multimodal_relational_continuity_resonance_architecture.md"
DESTINATION = ROOT / "spec" / "mrcra_evidence.json"
VERIFICATION_ARTIFACT = "outputs/mrcra_acceptance_manifest.json"
EMPIRICAL_IMPLEMENTATION = [
    "src/mrrn/empirical_acceptance.py",
    "outputs/mrcra_empirical_acceptance.json",
]
EMPIRICAL_TESTS = ["tests/test_empirical_acceptance.py"]
INTEGRATED_IMPLEMENTATION = [
    "src/mrrn/integrated_acceptance.py",
    "outputs/mrcra_integrated_acceptance.json",
]
INTEGRATED_TESTS = ["tests/test_integrated_acceptance.py"]
PERFORMANCE_IMPLEMENTATION = [
    "src/mrrn/performance_acceptance.py",
    "outputs/mrcra_performance_acceptance.json",
]
PERFORMANCE_TESTS = ["tests/test_performance_acceptance.py"]
INTEGRATED_MAJORS = {13, 14, 15, 16}

PATHS = {
    0: (["src/mrrn/config.py", "src/mrrn/cognitive_model.py"], ["tests/test_cognitive_foundation.py", "tests/test_cognitive_model.py"]),
    1: ([
        "src/mrrn/cognitive_types.py", "src/mrrn/provenance.py",
        "src/mrrn/boundaries.py", "src/mrrn/tensor_state.py",
    ], ["tests/test_cognitive_foundation.py", "tests/test_cognitive_foundation_v4.py", "tests/test_scoped_boundaries.py"]),
    2: ([
        "src/mrrn/cognitive_model.py", "src/mrrn/cognitive_checkpoint.py",
        "src/mrrn/reconstruction.py", "src/mrrn/abstraction_control.py",
        "src/mrrn/action_candidates.py", "src/mrrn/viability.py",
        "src/mrrn/evidence_requests.py", "src/mrrn/external_artifacts.py",
        "src/mrrn/metacognition.py",
    ], [
        "tests/test_cognitive_model.py", "tests/test_cognitive_checkpoint.py",
        "tests/test_cognitive_foundation_v4.py", "tests/test_evidence_artifacts.py",
        "tests/test_agent_session.py",
    ]),
    3: (["src/mrrn/cognitive_model.py", "src/mrrn/language.py"], ["tests/test_cognitive_model.py", "tests/test_cognitive_language.py"]),
    4: ([
        "src/mrrn/model.py", "src/mrrn/attention.py", "src/mrrn/mixer.py",
        "src/mrrn/lifting.py", "src/mrrn/scale_exchange.py",
        "src/mrrn/packed_projection.py",
    ], [
        "tests/test_model.py", "tests/test_extrapolation.py",
        "tests/test_attention.py", "tests/test_lifting.py",
        "tests/test_scale_exchange.py", "tests/test_packed_projection.py",
    ]),
    5: (["src/mrrn/modalities.py", "src/mrrn/observation.py", "src/mrrn/language.py"], ["tests/test_modalities.py", "tests/test_cognitive_foundation.py"]),
    6: (["src/mrrn/events.py"], ["tests/test_events.py"]),
    7: (["src/mrrn/relational_router.py", "src/mrrn/workspace.py"], ["tests/test_relational_router.py", "tests/test_workspace.py"]),
    8: (["src/mrrn/relational_router.py", "src/mrrn/workspace.py"], ["tests/test_relational_router.py", "tests/test_workspace.py"]),
    9: (["src/mrrn/workspace.py", "src/mrrn/cognitive_model.py"], ["tests/test_workspace.py", "tests/test_cognitive_model.py"]),
    10: (["src/mrrn/modalities.py", "src/mrrn/events.py", "src/mrrn/relational_router.py"], ["tests/test_modalities.py", "tests/test_events.py"]),
    11: (["src/mrrn/provenance.py", "src/mrrn/observation.py"], ["tests/test_cognitive_foundation.py"]),
    12: (["src/mrrn/memory_v2.py", "src/mrrn/cognitive_model.py"], ["tests/test_memory_v2.py", "tests/test_cognitive_reasoning.py"]),
    13: (["src/mrrn/compression.py", "src/mrrn/invariants.py"], ["tests/test_compression_invariants.py"]),
    14: (["src/mrrn/invariants.py"], ["tests/test_compression_invariants.py"]),
    15: (["src/mrrn/hypotheses.py", "src/mrrn/uncertainty.py", "src/mrrn/world_model.py"], ["tests/test_cognitive_reasoning.py", "tests/test_action_deliberation.py"]),
    16: (["src/mrrn/controller.py", "src/mrrn/cognitive_model.py", "src/mrrn/agent_session.py"], ["tests/test_cognitive_reasoning.py", "tests/test_cognitive_model.py", "tests/test_cognitive_actions.py", "tests/test_agent_session.py"]),
    17: ([
        "src/mrrn/cognitive_surprise.py", "src/mrrn/learning_progress.py",
        "src/mrrn/pc_rasl_acceptance.py",
        "outputs/pc_rasl_empirical_acceptance.json",
    ], [
        "tests/test_cognitive_surprise.py", "tests/test_learning_progress.py",
        "tests/test_pc_rasl_acceptance.py",
    ]),
    18: (["src/mrrn/cognitive_model.py", "src/mrrn/language.py", "src/mrrn/cognitive_diagnostics.py"], ["tests/test_cognitive_language.py", "tests/test_cognitive_model.py"]),
    19: (["src/mrrn/cognitive_objectives.py", "src/mrrn/cognitive_supervision.py", "src/mrrn/cognitive_training.py", "src/mrrn/training_profiles.py", "src/mrrn/gradient_governance.py", "scripts/train_fineweb.py", "scripts/train_mrcra_fineweb.py"], ["tests/test_cognitive_objectives.py", "tests/test_cognitive_supervision.py", "tests/test_cognitive_training.py", "tests/test_training_profiles.py", "tests/test_production_objectives.py", "tests/test_gradient_governance.py", "tests/test_fineweb_entrypoint.py"]),
    20: (["src/mrrn/cognitive_objectives.py", "src/mrrn/cognitive_supervision.py", "src/mrrn/cognitive_training.py", "src/mrrn/cognitive_surprise.py", "src/mrrn/learning_progress.py", "src/mrrn/pc_rasl_acceptance.py", "src/mrrn/continual_adaptation.py", "scripts/train_fineweb.py", "scripts/train_mrcra_fineweb.py", "scripts/run_pc_rasl_acceptance.py"], ["tests/test_cognitive_objectives.py", "tests/test_cognitive_supervision.py", "tests/test_cognitive_training.py", "tests/test_cognitive_surprise.py", "tests/test_learning_progress.py", "tests/test_pc_rasl_training.py", "tests/test_pc_rasl_acceptance.py", "tests/test_continual_adaptation.py", "tests/test_fineweb_entrypoint.py"]),
    21: (["src/mrrn/config.py", "src/mrrn/model.py", "src/mrrn/cognitive_training.py", "scripts/train_fineweb.py", "scripts/train_mrcra_fineweb.py"], ["tests/test_model.py", "tests/test_cognitive_training.py", "tests/test_fineweb_entrypoint.py"]),
    22: ([
        "src/mrrn/config.py", "src/mrrn/cognitive_model.py",
        "src/mrrn/language.py", "scripts/report_mrcra_parameters.py",
        "outputs/mrcra_1p3m_design_report.md",
        "outputs/mrcra_1p3m_parameter_report.json",
        "outputs/mrcra_8p4m_parameter_report.json",
        "outputs/mrcra_120m_parameter_report.json",
    ], [
        "tests/test_cognitive_foundation.py",
        "tests/test_fineweb_entrypoint.py",
    ]),
    23: ([
        "src/mrrn/attention.py", "src/mrrn/model.py",
        "src/mrrn/lifting.py", "src/mrrn/scale_exchange.py",
        "src/mrrn/packed_projection.py", "src/mrrn/cognitive_training.py",
        "scripts/train_fineweb.py", "scripts/train_mrcra_fineweb.py",
    ], [
        "tests/test_attention.py", "tests/test_model.py",
        "tests/test_lifting.py", "tests/test_scale_exchange.py",
        "tests/test_packed_projection.py", "tests/test_cognitive_training.py",
        "tests/test_fineweb_entrypoint.py",
    ]),
    24: (["src/mrrn/cognitive_model.py", "src/mrrn/cognitive_types.py", "src/mrrn/memory_v2.py"], ["tests/test_cognitive_model.py", "tests/test_cognitive_foundation.py"]),
    25: (["src/mrrn/cognitive_model.py", "src/mrrn/language.py"], ["tests/test_cognitive_model.py", "tests/test_cognitive_language.py"]),
    26: (["src/mrrn/cognitive_model.py", "src/mrrn/provenance.py", "src/mrrn/cognitive_training.py"], ["tests/test_cognitive_model.py", "tests/test_cognitive_training.py"]),
    27: (["src/mrrn/cognitive_types.py", "src/mrrn/invariants.py"], ["tests/test_compression_invariants.py"]),
    28: (["src/mrrn/traceability.py", "src/mrrn/cognitive_diagnostics.py"], ["tests/test_traceability.py", "tests/test_cognitive_model.py"]),
    29: (["src/mrrn/traceability.py", "spec/mrcra_evidence.json"], ["tests/test_traceability.py"]),
    30: (["outputs/multimodal_relational_continuity_resonance_architecture.md"], []),
    31: (["src/mrrn/cognitive_model.py", "src/mrrn/language.py"], ["tests/test_cognitive_model.py", "tests/test_cognitive_language.py"]),
}


def exact_test_nodes(paths: list[str]) -> list[str]:
    """Expand test modules to stable pytest node IDs at ledger-build time."""

    result: list[str] = []
    for relative in paths:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
        for item in tree.body:
            if (
                isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name.startswith("test_")
                # A proof cannot require its own prior result.  This meta-test
                # checks the completed artifact but is not evidence for itself.
                and item.name != "test_mrcra_verified_claims_have_exact_passing_nodeids_and_unchanged_sources"
            ):
                result.append(f"{relative}::{item.name}")
            elif isinstance(item, ast.ClassDef):
                for child in item.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test_"):
                        result.append(f"{relative}::{item.name}::{child.name}")
    return result


def entry(key: str, title: str) -> dict:
    if key.startswith("Gate "):
        letter = key[-1]
        if letter == "M":
            return {
                "status": "documented",
                "maturity": "contract",
                "claim": "Optional 32K CUDA hardware characterization is explicitly excluded from the current local acceptance scope by project direction.",
                "implementation": ["src/mrrn/cognitive_training.py", "scripts/benchmark_mrcra_32k.py"],
                "tests": exact_test_nodes(["tests/test_cognitive_training.py"]),
                "evidence_type": "scope_excluded_hardware_characterization",
            }
        if letter == "N":
            return {
                "status": "documented",
                "maturity": "integrated_loop",
                "claim": "Bounded production-path matched ablations and the fail-closed serious-checkpoint acceptance authority pass their contract tests, but matched end-task evidence still requires a trained serious checkpoint.",
                "implementation": [
                    "outputs/multimodal_relational_continuity_resonance_architecture.md",
                    "src/mrrn/serious_acceptance.py",
                    "scripts/run_mrcra_serious_checkpoint_acceptance.py",
                ] + INTEGRATED_IMPLEMENTATION,
                "tests": exact_test_nodes(["tests/test_serious_acceptance.py"]),
                "evidence_type": "trained_checkpoint_gate",
            }
        if letter in "GHIJKL":
            underlying = {
                "G": ["src/mrrn/memory_v2.py"],
                "H": ["src/mrrn/compression.py", "src/mrrn/knowledge.py"],
                "I": ["src/mrrn/uncertainty.py", "src/mrrn/hypotheses.py"],
                "J": ["src/mrrn/world_model.py"],
                "K": ["src/mrrn/controller.py"],
                "L": ["src/mrrn/surprise.py", "src/mrrn/cognitive_surprise.py"],
            }[letter]
            return {
                "status": "verified",
                "maturity": "mechanism",
                "claim": f"{title}; bounded learned-behavior evidence only, not serious-scale capability.",
                "implementation": underlying + EMPIRICAL_IMPLEMENTATION,
                "tests": exact_test_nodes(EMPIRICAL_TESTS),
                "evidence_type": "bounded_empirical_acceptance",
                "verification": {
                    "artifact": VERIFICATION_ARTIFACT,
                    "command_id": "python-tests",
                },
            }
        major = 28
    elif key.startswith("Stage "):
        stage = int(key.split()[1])
        major = 20
        if stage == 4:
            return {
                "status": "verified",
                "maturity": "mechanism",
                "claim": "Cross-modal event binding passes a bounded learned paired-versus-shuffled retrieval experiment.",
                "implementation": ["src/mrrn/modalities.py"] + EMPIRICAL_IMPLEMENTATION,
                "tests": exact_test_nodes(EMPIRICAL_TESTS),
                "evidence_type": "bounded_empirical_acceptance",
                "verification": {
                    "artifact": VERIFICATION_ARTIFACT,
                    "command_id": "python-tests",
                },
            }
        if stage == 9:
            return {
                "status": "verified",
                "maturity": "mechanism",
                "claim": "Replay, isolated adaptation, promotion, revocation, and exact rollback pass a bounded continual-learning experiment; live deployment remains gated.",
                "implementation": ["src/mrrn/knowledge.py"] + EMPIRICAL_IMPLEMENTATION,
                "tests": exact_test_nodes(EMPIRICAL_TESTS),
                "evidence_type": "bounded_empirical_acceptance",
                "verification": {
                    "artifact": VERIFICATION_ARTIFACT,
                    "command_id": "python-tests",
                },
            }
    else:
        major = int(key.split(".", 1)[0])
    implementation, test_files = PATHS[major]
    if major in INTEGRATED_MAJORS:
        implementation = implementation + INTEGRATED_IMPLEMENTATION
        test_files = test_files + INTEGRATED_TESTS
    if major == 23:
        implementation = implementation + PERFORMANCE_IMPLEMENTATION
        test_files = test_files + PERFORMANCE_TESTS
    tests = exact_test_nodes(test_files)
    status = "verified"
    if major in {27, 30}:
        status = "documented"
    return {
        "status": status,
        "maturity": "integrated_loop" if major in INTEGRATED_MAJORS else "contract",
        "claim": title or f"MRCRA requirement {key}",
        "implementation": implementation,
        "tests": tests,
        "evidence_type": "executable_contract" if status == "verified" else "design_contract",
        **({
            "verification": {
                "artifact": VERIFICATION_ARTIFACT,
                "command_id": "python-tests",
            }
        } if status == "verified" else {}),
    }


def main() -> None:
    sections = inventory(SPEC)
    evidence = {key: entry(key, title) for key, title in sections.items()}
    DESTINATION.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(evidence)} MRCRA evidence records to {DESTINATION}")


if __name__ == "__main__":
    main()
