from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_mrcra_acceptance", ROOT / "scripts" / "run_mrcra_acceptance.py"
)
assert SPEC is not None and SPEC.loader is not None
acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceptance)


def test_pytest_result_parser_preserves_parameter_ids_with_spaces():
    line = (
        "tests/test_example.py::test_contract[change0-full packed context] "
        "PASSED [100%]"
    )
    match = acceptance.RESULT.match(line)
    assert match is not None
    assert match.group(2) == "PASSED"
    assert acceptance.normalized_node(match.group(1)) == (
        "tests/test_example.py::test_contract"
    )


def test_node_normalization_handles_nested_parameter_renderings():
    assert acceptance.normalized_node(
        "tests/test_example.py::test_contract[value[1] and label]"
    ) == "tests/test_example.py::test_contract"


def test_retained_acceptance_text_omits_local_checkout_and_home_paths():
    local = acceptance.ROOT / "outputs" / "artifact.json"
    home = Path.home() / ".nvm" / "bin" / "node"
    assert acceptance.portable_text(str(local)) == "./outputs/artifact.json"
    assert acceptance.portable_text(str(home)) == "~/.nvm/bin/node"


def test_source_inventory_excludes_transient_machine_artifacts():
    transient = (
        Path("src/.DS_Store"),
        Path("src/mrrn/__pycache__/model.cpython-311.pyc"),
        Path("tests/.pytest_cache/state"),
        Path("scripts/verify_spec.py~"),
        Path("spec/mrcra_evidence.json.tmp"),
    )
    assert all(not acceptance.is_durable_source(path) for path in transient)
    assert acceptance.is_durable_source(Path("src/mrrn/model.py"))
    assert acceptance.is_durable_source(Path("trackio_frontend/src/App.svelte"))
