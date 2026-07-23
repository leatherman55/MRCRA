"""Append-only authoritative provenance DAG for MRCRA.

Records never change after insertion.  Verification changes are separate ledger
events, so revocation can propagate without rewriting the historical source or
derivation record that downstream neural state references.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Iterable, Mapping

import torch
from torch import Tensor

from .cognitive_types import ModalityClass, SourceClass, SupportInterval, VerificationClass


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    record_id: int
    source_class: SourceClass
    source_uri_or_episode: str
    support: SupportInterval
    modality: ModalityClass
    parents: tuple[int, ...]
    operator: str
    scenario_id: int
    model_authority: str
    verification_at_creation: VerificationClass
    source_reliability: float
    calibration_context: str
    spatial_region: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.record_id < 0 or self.scenario_id < 0:
            raise ValueError("record and scenario IDs cannot be negative")
        if not self.source_uri_or_episode or not self.operator or not self.model_authority:
            raise ValueError("source, operator, and model authority must be nonempty")
        if not 0 <= self.source_reliability <= 1:
            raise ValueError("source reliability must lie in [0,1]")
        if len(set(self.parents)) != len(self.parents) or any(parent < 0 for parent in self.parents):
            raise ValueError("parent IDs must be unique and nonnegative")
        if self.record_id in self.parents:
            raise ValueError("a provenance record cannot parent itself")


@dataclass(frozen=True, slots=True)
class VerificationEvent:
    sequence: int
    record_id: int
    state: VerificationClass
    authority: str
    reason: str

    def __post_init__(self) -> None:
        if self.sequence < 0 or self.record_id < 0:
            raise ValueError("verification event IDs cannot be negative")
        if not self.authority or not self.reason:
            raise ValueError("verification authority and reason must be nonempty")


class ProvenanceLedger:
    """In-memory correctness authority with deterministic portable state.

    Production deployments may mirror this API to an external append-only
    database.  Inference tensors store only IDs and bounded learned features.
    """

    FORMAT_VERSION = 1

    def __init__(self) -> None:
        self._records: list[ProvenanceRecord] = []
        self._verification_events: list[VerificationEvent] = []
        self._children: dict[int, set[int]] = {}

    def __len__(self) -> int:
        return len(self._records)

    @property
    def verification_event_count(self) -> int:
        return len(self._verification_events)

    def get(self, record_id: int) -> ProvenanceRecord:
        if not 0 <= record_id < len(self._records):
            raise KeyError(f"unknown provenance record {record_id}")
        return self._records[record_id]

    def records(self) -> tuple[ProvenanceRecord, ...]:
        return tuple(self._records)

    def verification_events(self) -> tuple[VerificationEvent, ...]:
        return tuple(self._verification_events)

    @staticmethod
    def _normalize_parents(parents: Iterable[int]) -> tuple[int, ...]:
        return tuple(dict.fromkeys(int(parent) for parent in parents))

    def append(
        self,
        *,
        source_class: SourceClass,
        source_uri_or_episode: str,
        support: SupportInterval,
        modality: ModalityClass,
        parents: Iterable[int] = (),
        operator: str,
        scenario_id: int,
        model_authority: str,
        verification: VerificationClass = VerificationClass.UNVERIFIED,
        source_reliability: float = 1.0,
        calibration_context: str = "unspecified",
        spatial_region: Iterable[float] = (),
    ) -> int:
        """Append one record after enforcing causal parent and source rules."""

        normalized = self._normalize_parents(parents)
        record_id = len(self._records)
        if any(parent >= record_id or parent < 0 for parent in normalized):
            raise ValueError("parents must reference earlier records in this ledger")
        if normalized and source_class in (SourceClass.EXTERNAL, SourceClass.BODILY):
            raise ValueError("derived records cannot claim an authoritative observed source class")
        if not normalized and source_class in (
            SourceClass.RETRIEVED, SourceClass.INFERRED, SourceClass.PREDICTED,
            SourceClass.SIMULATED, SourceClass.ABSTRACTED,
            SourceClass.RECONSTRUCTED, SourceClass.TOOL_OUTPUT,
            SourceClass.COMMUNICATED, SourceClass.EXTERNAL_ARTIFACT,
        ):
            raise ValueError("derived source classes require at least one parent")
        if source_class == SourceClass.SIMULATED and scenario_id == 0:
            raise ValueError("simulated records require a nonzero scenario ID")
        if normalized:
            parent_scenarios = {self.get(parent).scenario_id for parent in normalized}
            if source_class != SourceClass.SIMULATED and len(parent_scenarios - {0, scenario_id}) > 0:
                raise ValueError("derived record cannot silently cross scenario boundaries")
        record = ProvenanceRecord(
            record_id, SourceClass(source_class), source_uri_or_episode, support,
            ModalityClass(modality), normalized, operator, scenario_id,
            model_authority, VerificationClass(verification), float(source_reliability),
            calibration_context, tuple(float(value) for value in spatial_region),
        )
        self._records.append(record)
        self._children[record_id] = set()
        for parent in normalized:
            self._children[parent].add(record_id)
        return record_id

    def derive(
        self, parents: Iterable[int], *, source_class: SourceClass, operator: str,
        support: SupportInterval, modality: ModalityClass, scenario_id: int = 0,
        model_authority: str, source_reliability: float | None = None,
        calibration_context: str = "derived",
    ) -> int:
        normalized = self._normalize_parents(parents)
        if not normalized:
            raise ValueError("a derivation requires parent records")
        authorities = sorted({self.get(parent).source_uri_or_episode for parent in normalized})
        reliability = (
            min(self.get(parent).source_reliability for parent in normalized)
            if source_reliability is None else source_reliability
        )
        return self.append(
            source_class=source_class,
            source_uri_or_episode="derived:" + sha256("\0".join(authorities).encode()).hexdigest()[:16],
            support=support,
            modality=modality,
            parents=normalized,
            operator=operator,
            scenario_id=scenario_id,
            model_authority=model_authority,
            source_reliability=reliability,
            calibration_context=calibration_context,
        )

    def set_verification(
        self, record_id: int, state: VerificationClass, *, authority: str, reason: str,
    ) -> None:
        self.get(record_id)
        state = VerificationClass(state)
        current = self.effective_verification(record_id, propagate_revocation=False)
        if current == VerificationClass.REVOKED and state != VerificationClass.REVOKED:
            raise ValueError("revocation is terminal; append a new provenance record instead")
        self._verification_events.append(
            VerificationEvent(len(self._verification_events), record_id, state, authority, reason)
        )

    def effective_verification(
        self, record_id: int, *, propagate_revocation: bool = True,
    ) -> VerificationClass:
        record = self.get(record_id)
        state = record.verification_at_creation
        for event in self._verification_events:
            if event.record_id == record_id:
                state = event.state
        if propagate_revocation and state != VerificationClass.REVOKED:
            if any(
                self.effective_verification(parent, propagate_revocation=True)
                == VerificationClass.REVOKED
                for parent in record.parents
            ):
                return VerificationClass.REVOKED
        return state

    def descendants(self, record_id: int) -> tuple[int, ...]:
        self.get(record_id)
        visited: set[int] = set()
        frontier = list(self._children[record_id])
        while frontier:
            child = frontier.pop()
            if child not in visited:
                visited.add(child)
                frontier.extend(self._children[child])
        return tuple(sorted(visited))

    def lineage(self, record_id: int) -> tuple[int, ...]:
        self.get(record_id)
        visited: set[int] = set()
        frontier = list(self.get(record_id).parents)
        while frontier:
            parent = frontier.pop()
            if parent not in visited:
                visited.add(parent)
                frontier.extend(self.get(parent).parents)
        return tuple(sorted(visited))

    def independent_roots(self, record_id: int) -> tuple[int, ...]:
        """Return unique source roots; duplicate derivations do not add evidence."""

        record = self.get(record_id)
        if not record.parents:
            return (record_id,)
        roots: set[int] = set()
        frontier = list(record.parents)
        while frontier:
            current = self.get(frontier.pop())
            if current.parents:
                frontier.extend(current.parents)
            else:
                roots.add(current.record_id)
        return tuple(sorted(roots))

    def can_consolidate(self, record_id: int) -> bool:
        record = self.get(record_id)
        return (
            record.source_class not in (SourceClass.SIMULATED, SourceClass.PREDICTED)
            and self.effective_verification(record_id) in (
                VerificationClass.INTERNALLY_CONSISTENT,
                VerificationClass.EXTERNALLY_CHECKED,
            )
        )

    def can_justify_external_action(self, record_id: int) -> bool:
        record = self.get(record_id)
        return (
            record.source_class != SourceClass.SIMULATED
            and self.effective_verification(record_id) not in (
                VerificationClass.CONTRADICTED, VerificationClass.REVOKED,
            )
        )

    def feature_vector(
        self, record_id: int, width: int, *, device=None, dtype=None,
    ) -> Tensor:
        """Return the fixed differentiable view of immutable source metadata.

        The returned tensor is an input feature, never the authority.  Its first
        eight channels encode ``SourceClass``; the next five encode effective
        verification; the remaining canonical channels are source reliability,
        a nonzero-scenario flag, and a derived-source flag.  Wider contracts are
        zero padded and narrower contracts are deterministically truncated.
        """

        if width <= 0:
            raise ValueError("provenance feature width must be positive")
        record = self.get(record_id)
        canonical = torch.zeros(
            len(SourceClass) + len(VerificationClass) + 3,
            device=device, dtype=dtype,
        )
        canonical[int(record.source_class)] = 1
        verification_offset = len(SourceClass)
        canonical[
            verification_offset + int(self.effective_verification(record_id))
        ] = 1
        canonical[-3] = record.source_reliability
        canonical[-2] = float(record.scenario_id != 0)
        canonical[-1] = float(bool(record.parents))
        if width <= canonical.numel():
            return canonical[:width]
        return torch.cat((
            canonical,
            torch.zeros(width - canonical.numel(), device=device, dtype=dtype),
        ))

    def state_dict(self) -> dict:
        def record_dict(record: ProvenanceRecord) -> dict:
            result = asdict(record)
            result["source_class"] = int(record.source_class)
            result["modality"] = int(record.modality)
            result["verification_at_creation"] = int(record.verification_at_creation)
            return result

        return {
            "format_version": self.FORMAT_VERSION,
            "records": [record_dict(record) for record in self._records],
            "verification_events": [
                {
                    "sequence": event.sequence,
                    "record_id": event.record_id,
                    "state": int(event.state),
                    "authority": event.authority,
                    "reason": event.reason,
                }
                for event in self._verification_events
            ],
        }

    def load_state_dict(self, state: Mapping) -> None:
        if state.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("unsupported provenance format version")
        rebuilt = ProvenanceLedger()
        for raw in state.get("records", []):
            support = raw["support"]
            actual_id = rebuilt.append(
                source_class=SourceClass(raw["source_class"]),
                source_uri_or_episode=raw["source_uri_or_episode"],
                support=SupportInterval(**support),
                modality=ModalityClass(raw["modality"]),
                parents=raw["parents"],
                operator=raw["operator"],
                scenario_id=raw["scenario_id"],
                model_authority=raw["model_authority"],
                verification=VerificationClass(raw["verification_at_creation"]),
                source_reliability=raw["source_reliability"],
                calibration_context=raw["calibration_context"],
                spatial_region=raw.get("spatial_region", ()),
            )
            if actual_id != raw["record_id"]:
                raise ValueError("provenance record IDs are not contiguous")
        for raw in state.get("verification_events", []):
            if raw["sequence"] != rebuilt.verification_event_count:
                raise ValueError("verification event sequence is not contiguous")
            rebuilt.set_verification(
                raw["record_id"], VerificationClass(raw["state"]),
                authority=raw["authority"], reason=raw["reason"],
            )
        self._records = rebuilt._records
        self._verification_events = rebuilt._verification_events
        self._children = rebuilt._children

    def digest(self) -> str:
        payload = json.dumps(self.state_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()
