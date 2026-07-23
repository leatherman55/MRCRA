import json
import os
from pathlib import Path

import pytest

from mrrn.traceability import audit, inventory

ROOT = Path(__file__).resolve().parents[1]


def test_current_evidence_is_valid_and_inventory_covers_complete_specification():
    sections = inventory(ROOT / "outputs" / "multiresolution_resonance_network_spec.md")
    assert {str(number) for number in range(36)} <= sections.keys()
    assert {f"Gate {letter}" for letter in "ABCDEFGHI"} <= sections.keys()
    assert {f"Stage {number}" for number in range(7)} <= sections.keys()
    report = audit(ROOT)
    assert not report.invalid
    assert report.total == len(sections)
    assert report.evidenced == report.verified == report.total == 189
    assert report.complete
    assert audit(ROOT, strict=True).complete


def test_mrcra_evidence_covers_every_heading_and_records_cuda_as_scope_excluded():
    specification = "outputs/multimodal_relational_continuity_resonance_architecture.md"
    evidence = "spec/mrcra_evidence.json"
    sections = inventory(ROOT / specification)
    assert {str(number) for number in range(32)} <= sections.keys()
    assert {f"Gate {letter}" for letter in "ABCDEFGHIJKLMN"} <= sections.keys()
    assert {f"Stage {number}" for number in range(10)} <= sections.keys()
    report = audit(ROOT, specification=specification, evidence_file=evidence)
    assert report.evidenced_complete
    assert report.total == report.evidenced == 174
    assert report.external == 0
    assert report.documented == 4
    maturity = dict(report.maturity_counts)
    assert maturity["mechanism"] > 0
    assert maturity["integrated_loop"] > 0
    assert maturity["serious_checkpoint"] == 0
    assert maturity["deployment"] == 0
    assert not report.complete
    strict = audit(
        ROOT, strict=True, specification=specification, evidence_file=evidence
    )
    assert not strict.complete
    assert any("Gate M: strict audit requires verified" in item for item in strict.invalid)


def test_mrcra_verified_claims_have_exact_passing_nodeids_and_unchanged_sources():
    if os.environ.get("MRCRA_BUILDING_ACCEPTANCE") == "1":
        pytest.skip("the hash-bound artifact is being replaced by this run")
    report = audit(
        ROOT, executable=True,
        specification="outputs/multimodal_relational_continuity_resonance_architecture.md",
        evidence_file="spec/mrcra_evidence.json",
    )
    assert not report.invalid
    assert report.verified == 170


def test_inventory_and_evidence_validation_failure_paths(tmp_path):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "spec").mkdir()
    spec = tmp_path / "outputs" / "multiresolution_resonance_network_spec.md"
    evidence = tmp_path / "spec" / "evidence.json"
    spec.write_text("## no keys", encoding="utf-8")
    with pytest.raises(ValueError, match="no traceable"):
        inventory(spec)
    spec.write_text("## 0. First\n## 0. Duplicate", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        inventory(spec)

    spec.write_text("## 0. First\n## 1. Second\n## 2. Third\n## 3. Fourth", encoding="utf-8")
    (tmp_path / "exists.py").write_text("pass\n")
    evidence.write_text(
        json.dumps(
            {
                "0": {"status": "wrong", "maturity": "imaginary", "claim": "", "implementation": [], "tests": []},
                "1": {"status": "verified"},
                "2": {"status": "verified", "maturity": "contract", "claim": "claim", "implementation": [], "tests": []},
                "3": {"status": "documented", "maturity": "contract", "claim": "claim", "implementation": ["missing.py"], "tests": ["exists.py"]},
                "unknown": {"status": "verified", "maturity": "contract"},
            }
        ),
        encoding="utf-8",
    )
    report = audit(tmp_path, strict=True)
    assert report.missing == ()
    assert any("invalid status" in item for item in report.invalid)
    assert any("invalid maturity" in item for item in report.invalid)
    assert any("empty claim" in item for item in report.invalid)
    assert any("unknown section" in item for item in report.invalid)
    assert any("missing fields" in item for item in report.invalid)
    assert any("verified without" in item for item in report.invalid)
    assert any("missing implementation path" in item for item in report.invalid)
    assert any("strict audit" in item for item in report.invalid)
