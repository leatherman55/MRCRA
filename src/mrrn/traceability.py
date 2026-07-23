"""Machine-checked mapping from every specification heading to empirical evidence."""

from __future__ import annotations

import json
import re
import ast
from hashlib import sha256
from dataclasses import dataclass
from pathlib import Path

HEADING = re.compile(
    r"^#{2,3} (?P<key>\d+(?:\.\d+)?|Gate [A-Z]|Stage \d+)(?:[.:] )?(?P<title>.*)$"
)
VALID_STATUSES = {"verified", "documented", "external"}
VALID_MATURITIES = {
    "contract", "mechanism", "integrated_loop", "transfer",
    "serious_checkpoint", "deployment",
}


@dataclass(frozen=True, slots=True)
class TraceabilityReport:
    total: int
    evidenced: int
    verified: int
    documented: int
    external: int
    missing: tuple[str, ...]
    invalid: tuple[str, ...]
    maturity_counts: tuple[tuple[str, int], ...]

    @property
    def complete(self) -> bool:
        return not self.missing and not self.invalid and self.verified == self.total

    @property
    def evidenced_complete(self) -> bool:
        return not self.missing and not self.invalid and self.evidenced == self.total


def inventory(specification: Path) -> dict[str, str]:
    """Extract every numbered section, stage, and verification gate."""

    result: dict[str, str] = {}
    for line in specification.read_text(encoding="utf-8").splitlines():
        match = HEADING.match(line)
        if match:
            key, title = match.group("key"), match.group("title").strip()
            if key in result:
                raise ValueError(f"duplicate specification key {key}")
            result[key] = title
    if not result:
        raise ValueError("specification contains no traceable headings")
    return result


def audit(
    root: Path, *, strict: bool = False,
    specification: str | Path = "outputs/multiresolution_resonance_network_spec.md",
    evidence_file: str | Path = "spec/evidence.json",
    executable: bool = False,
) -> TraceabilityReport:
    """Validate evidence paths and optionally require all sections to be verified."""

    specification = Path(specification)
    evidence_file = Path(evidence_file)
    resolved_evidence = evidence_file if evidence_file.is_absolute() else root / evidence_file
    sections = inventory(specification if specification.is_absolute() else root / specification)
    evidence = json.loads(
        resolved_evidence.read_text(encoding="utf-8")
    )
    invalid: list[str] = []
    node_cache: dict[Path, set[str]] = {}

    def test_nodes(path: Path) -> set[str]:
        if path not in node_cache:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            nodes: set[str] = set()
            for item in tree.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test_"):
                    nodes.add(item.name)
                elif isinstance(item, ast.ClassDef):
                    for child in item.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test_"):
                            nodes.add(f"{item.name}::{child.name}")
            node_cache[path] = nodes
        return node_cache[path]

    for key, item in evidence.items():
        if key not in sections:
            invalid.append(f"unknown section {key}")
            continue
        missing_fields = {"status", "maturity", "claim", "implementation", "tests"} - item.keys()
        if missing_fields:
            invalid.append(f"{key}: missing fields {sorted(missing_fields)}")
            continue
        if item["status"] not in VALID_STATUSES:
            invalid.append(f"{key}: invalid status {item['status']}")
        if item["maturity"] not in VALID_MATURITIES:
            invalid.append(f"{key}: invalid maturity {item['maturity']}")
        if not item["claim"].strip():
            invalid.append(f"{key}: empty claim")
        for category in ("implementation", "tests"):
            if item["status"] == "verified" and not item[category]:
                invalid.append(f"{key}: verified without {category}")
            for relative in item[category]:
                path = root / relative.split("::", 1)[0]
                if not path.is_file():
                    invalid.append(f"{key}: missing {category} path {relative}")
                    continue
                if executable and category == "tests":
                    parts = relative.split("::", 1)
                    if len(parts) != 2 or parts[1] not in test_nodes(path):
                        invalid.append(f"{key}: test is not an exact collected node {relative}")
        if executable and item.get("status") == "verified":
            verification = item.get("verification")
            if not isinstance(verification, dict):
                invalid.append(f"{key}: verified without an executable verification record")
                continue
            artifact_name = verification.get("artifact")
            command_id = verification.get("command_id")
            if not artifact_name or not command_id:
                invalid.append(f"{key}: incomplete executable verification record")
                continue
            artifact_path = root / artifact_name
            if not artifact_path.is_file():
                invalid.append(f"{key}: missing verification artifact {artifact_name}")
                continue
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            commands = {command.get("id"): command for command in artifact.get("commands", [])}
            command = commands.get(command_id)
            if not command or command.get("exit_code") != 0:
                invalid.append(f"{key}: verification command {command_id!r} did not pass")
                continue
            passed = set(command.get("passed_nodeids", []))
            missing_nodes = sorted(set(item["tests"]) - passed)
            if missing_nodes:
                invalid.append(f"{key}: verification artifact omitted tests {missing_nodes}")
            source_hashes = artifact.get("source_sha256", {})
            for implementation in item["implementation"]:
                relative = implementation.split("::", 1)[0]
                path = root / relative
                # The ledger cannot contain a prior hash of itself.  Its schema,
                # paths, node IDs, and command references are checked directly.
                if path.resolve() == resolved_evidence.resolve():
                    continue
                if path.is_file() and relative in source_hashes:
                    digest = sha256(path.read_bytes()).hexdigest()
                    if source_hashes[relative] != digest:
                        invalid.append(f"{key}: implementation changed after verification: {relative}")
    missing = tuple(key for key in sections if key not in evidence)
    verified = sum(item.get("status") == "verified" for item in evidence.values())
    documented = sum(item.get("status") == "documented" for item in evidence.values())
    external = sum(item.get("status") == "external" for item in evidence.values())
    maturity_counts = tuple(
        (maturity, sum(item.get("maturity") == maturity for item in evidence.values()))
        for maturity in sorted(VALID_MATURITIES)
    )
    if strict:
        for key, item in evidence.items():
            if item.get("status") != "verified":
                invalid.append(f"{key}: strict audit requires verified status")
    return TraceabilityReport(
        total=len(sections),
        evidenced=len(evidence),
        verified=verified, documented=documented, external=external,
        missing=missing,
        invalid=tuple(invalid),
        maturity_counts=maturity_counts,
    )
