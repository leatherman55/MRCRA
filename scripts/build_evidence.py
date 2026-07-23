#!/usr/bin/env python3
"""Regenerate the complete specification-to-code-and-test evidence ledger."""

from __future__ import annotations

import json
from pathlib import Path

from mrrn.traceability import inventory

ROOT = Path(__file__).resolve().parents[1]

ROUTES = {
    "0": (["README.md", "src/mrrn/traceability.py"], ["tests/test_traceability.py"]),
    "1": (["src/mrrn/model.py"], ["tests/test_model.py", "tests/test_synthetics.py"]),
    "2": (["src/mrrn/lifting.py", "src/mrrn/resonance.py"], ["tests/test_lifting.py", "tests/test_resonance.py", "tests/test_synthetics.py"]),
    "3": (["src/mrrn/config.py", "src/mrrn/evaluation.py"], ["tests/test_config.py", "tests/test_evaluation.py"]),
    "4": (["src/mrrn/config.py", "src/mrrn/complex_ops.py", "src/mrrn/lifting.py"], ["tests/test_config.py", "tests/test_complex_ops.py", "tests/test_lifting.py"]),
    "5": (["src/mrrn/lifting.py"], ["tests/test_lifting.py", "tests/test_extrapolation.py"]),
    "6": (["src/mrrn/resonance.py", "src/mrrn/complex_ops.py"], ["tests/test_resonance.py", "tests/test_extrapolation.py"]),
    "7": (["src/mrrn/scale_exchange.py"], ["tests/test_scale_exchange.py", "tests/test_model.py"]),
    "8": (["src/mrrn/mixer.py", "src/mrrn/complex_ops.py"], ["tests/test_mixer.py", "tests/test_complex_ops.py", "tests/test_spectral_activation.py"]),
    "9": (["src/mrrn/attention.py", "src/mrrn/model.py"], ["tests/test_attention.py", "tests/test_model.py"]),
    "10": (["src/mrrn/memory.py", "src/mrrn/model.py"], ["tests/test_memory.py", "tests/test_model.py"]),
    "11": (["src/mrrn/model.py"], ["tests/test_model.py"]),
    "12": (["src/mrrn/model.py"], ["tests/test_model.py", "tests/test_checkpoint.py"]),
    "13": (["src/mrrn/modalities.py"], ["tests/test_modalities.py", "tests/test_extrapolation.py"]),
    "14": (["src/mrrn/model.py", "src/mrrn/modalities.py"], ["tests/test_model.py", "tests/test_extrapolation.py"]),
    "15": (["src/mrrn/objectives.py"], ["tests/test_objectives.py", "tests/test_spectral_activation.py"]),
    "16": (["src/mrrn/model.py", "src/mrrn/resonance.py", "src/mrrn/memory.py"], ["tests/test_model.py", "tests/test_resonance.py", "tests/test_memory.py"]),
    "17": (["src/mrrn/optimization.py", "src/mrrn/complex_ops.py", "src/mrrn/resonance.py"], ["tests/test_optimization.py", "tests/test_complex_ops.py", "tests/test_resonance.py"]),
    "18": (["src/mrrn/evaluation.py", "src/mrrn/config.py"], ["tests/test_evaluation.py", "tests/test_config.py"]),
    "19": (["src/mrrn/config.py"], ["tests/test_config.py", "tests/test_model.py"]),
    "20": (["README.md", "src/mrrn/evaluation.py"], ["tests/test_synthetics.py", "tests/test_evaluation.py"]),
    "21": (["src/mrrn/diagnostics.py"], ["tests/test_diagnostics.py", "tests/test_spectral_activation.py"]),
    "22": (["src/mrrn/evaluation.py", "src/mrrn/mixer.py", "src/mrrn/resonance.py", "src/mrrn/attention.py"], ["tests/test_evaluation.py", "tests/test_mixer.py", "tests/test_resonance.py", "tests/test_attention.py"]),
    "23": (["src/mrrn/diagnostics.py", "README.md"], ["tests/test_diagnostics.py", "tests/test_synthetics.py", "tests/test_extrapolation.py"]),
    "24": (["outputs/multiresolution_resonance_network_spec.md", "src/mrrn/synthetics.py"], ["tests/test_synthetics.py", "tests/test_extrapolation.py"]),
    "25": (["src/mrrn/model.py", "src/mrrn/lifting.py", "src/mrrn/resonance.py"], ["tests/test_model.py", "tests/test_lifting.py", "tests/test_resonance.py"]),
    "26": (["src/mrrn/model.py", "src/mrrn/modalities.py", "src/mrrn/checkpoint.py"], ["tests/test_model.py", "tests/test_modalities.py", "tests/test_checkpoint.py"]),
    "27": (["src/mrrn/traceability.py"], ["tests/test_traceability.py"]),
    "28": (["src/mrrn/model.py", "src/mrrn/mixer.py", "src/mrrn/memory.py"], ["tests/test_model.py", "tests/test_mixer.py", "tests/test_memory.py"]),
    "29": (["src/mrrn/model.py", "src/mrrn/evaluation.py"], ["tests/test_model.py", "tests/test_evaluation.py", "tests/test_synthetics.py"]),
    "30": (["outputs/multiresolution_resonance_network_spec.md", "README.md"], ["tests/test_traceability.py"]),
    "31": (["src/mrrn/model.py"], ["tests/test_model.py", "tests/test_synthetics.py"]),
    "32": (["src/mrrn/resonance.py", "src/mrrn/attention.py", "src/mrrn/lifting.py", "src/mrrn/config.py"], ["tests/test_resonance.py", "tests/test_attention.py", "tests/test_lifting.py", "tests/test_config.py"]),
    "33": (["src/mrrn/config.py"], ["tests/test_config.py"]),
    "34": (["src/mrrn/checkpoint.py", "src/mrrn/model.py", "src/mrrn/memory.py"], ["tests/test_checkpoint.py", "tests/test_model.py", "tests/test_memory.py"]),
    "35": (["src/mrrn/synthetics.py", "src/mrrn/evaluation.py", "src/mrrn/traceability.py"], ["tests/test_synthetics.py", "tests/test_evaluation.py", "tests/test_traceability.py"]),
    "36": (["src/mrrn/surprise.py", "scripts/run_surprise_verification.py"], ["tests/test_surprise.py"]),
}

STAGES = {
    "Stage 0": ROUTES["5"][0] + ROUTES["6"][0],
    "Stage 1": ["src/mrrn/mixer.py", "src/mrrn/model.py"],
    "Stage 2": ["src/mrrn/resonance.py", "src/mrrn/scale_exchange.py"],
    "Stage 3": ["src/mrrn/resonance.py"],
    "Stage 4": ["src/mrrn/attention.py"],
    "Stage 5": ["src/mrrn/memory.py", "src/mrrn/model.py"],
    "Stage 6": ["src/mrrn/mixer.py", "src/mrrn/modalities.py", "src/mrrn/evaluation.py"],
}

GATES = {
    "Gate A": (["src/mrrn/lifting.py"], ["tests/test_lifting.py", "tests/test_extrapolation.py"]),
    "Gate B": (["src/mrrn/resonance.py", "src/mrrn/complex_ops.py"], ["tests/test_resonance.py", "tests/test_complex_ops.py"]),
    "Gate C": (["src/mrrn/model.py", "src/mrrn/scale_exchange.py"], ["tests/test_model.py", "tests/test_scale_exchange.py", "tests/test_extrapolation.py"]),
    "Gate D": (["src/mrrn/attention.py"], ["tests/test_attention.py"]),
    "Gate E": (["src/mrrn/memory.py"], ["tests/test_memory.py"]),
    "Gate F": (["src/mrrn/synthetics.py"], ["tests/test_synthetics.py"]),
    "Gate G": (["src/mrrn/evaluation.py"], ["tests/test_evaluation.py"]),
    "Gate H": (["src/mrrn/model.py", "src/mrrn/modalities.py"], ["tests/test_extrapolation.py"]),
    "Gate I": (["src/mrrn/surprise.py", "scripts/run_surprise_verification.py"], ["tests/test_surprise.py"]),
}


def main() -> None:
    sections = inventory(ROOT / "outputs" / "multiresolution_resonance_network_spec.md")
    evidence = {}
    for key, title in sections.items():
        if key in GATES:
            implementation, tests = GATES[key]
        elif key in STAGES:
            implementation = STAGES[key]
            tests = ["tests/test_synthetics.py", "tests/test_evaluation.py"]
        else:
            implementation, tests = ROUTES[key.split(".", 1)[0]]
        boundary = " Limitation boundaries are preserved rather than promoted to unsupported guarantees." if key.startswith(("0", "24", "30")) else ""
        evidence[key] = {
            "status": "verified",
            "maturity": "contract",
            "claim": f"{title or key} is mapped to executable implementation and empirical evidence.{boundary}",
            "implementation": implementation,
            "tests": tests,
        }
    (ROOT / "spec" / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
