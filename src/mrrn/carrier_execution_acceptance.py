"""Fail-closed empirical acceptance for optimized carrier execution.

This suite tests mechanisms, not target-hardware throughput.  It measures the
exact forward/adjoint error of both custom composite operations, concrete
autograd tensor retention, coarse-checkpoint state continuity, and the
document planner's target bijection/invocation reduction.  A failure in any
semantic or materialization gate makes the complete report fail.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import torch
from torch import Tensor

from .carrier_execution import fused_simplex_residual
from .config import MRRNConfig
from .document_batching import DocumentMajorBatchPlanner
from .lm_training import (
    ByteTextTokenizer,
    PackedTokenStream,
    SequenceTextSource,
)
from .model import MRRN
from .resonance import associative_affine_scan


@dataclass(frozen=True, slots=True)
class CarrierExecutionCriterion:
    name: str
    measurement: float
    threshold: float
    direction: str
    unit: str
    passed: bool


@dataclass(frozen=True, slots=True)
class CarrierExecutionAcceptanceReport:
    format_version: int
    suite: str
    device: str
    dtype: str
    criteria: tuple[CarrierExecutionCriterion, ...]
    telemetry: dict[str, float | int | str | bool]
    passed: bool
    claim_boundary: str

    def to_dict(self) -> dict:
        return asdict(self)


def _criterion(
    name: str,
    measurement: float,
    threshold: float,
    direction: str,
    unit: str,
) -> CarrierExecutionCriterion:
    if direction == "maximum":
        passed = measurement <= threshold
    elif direction == "minimum":
        passed = measurement >= threshold
    else:
        raise ValueError("acceptance direction must be maximum or minimum")
    return CarrierExecutionCriterion(
        name, measurement, threshold, direction, unit, passed
    )


def _graph_nodes(value: Tensor) -> int:
    pending = [value.grad_fn]
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        pending.extend(parent for parent, _ in node.next_functions)
    return len(seen)


def _scan_evidence() -> tuple[tuple[CarrierExecutionCriterion, ...], dict]:
    torch.manual_seed(1601)
    shape = (2, 63, 2, 3, 2)
    transition = torch.randn(shape, dtype=torch.float64) * 0.08
    transition[..., 0] += 0.84
    drive = torch.randn(shape, dtype=torch.float64) * 0.04
    initial = torch.randn(2, 2, 3, 2, dtype=torch.float64) * 0.1
    cotangent = torch.randn(shape, dtype=torch.float64)

    reference_inputs = tuple(
        value.detach().clone().requires_grad_(True)
        for value in (transition, drive, initial)
    )
    optimized_inputs = tuple(
        value.detach().clone().requires_grad_(True)
        for value in (transition, drive, initial)
    )

    def retained(implementation: str, inputs):
        saved: list[int] = []

        def pack(value: Tensor):
            saved.append(value.numel() * value.element_size())
            return value

        started = perf_counter()
        with torch.autograd.graph.saved_tensors_hooks(
            pack, lambda value: value
        ):
            output = associative_affine_scan(
                *inputs, implementation=implementation
            )
        elapsed = perf_counter() - started
        return output, len(saved), sum(saved), elapsed

    reference, reference_count, reference_bytes, reference_seconds = retained(
        "composite", reference_inputs
    )
    optimized, optimized_count, optimized_bytes, optimized_seconds = retained(
        "custom_adjoint", optimized_inputs
    )
    reference_gradients = torch.autograd.grad(
        reference, reference_inputs, cotangent
    )
    optimized_gradients = torch.autograd.grad(
        optimized, optimized_inputs, cotangent
    )
    forward_error = float((optimized - reference).abs().max().detach())
    gradient_error = max(
        float((actual - expected).abs().max().detach())
        for actual, expected in zip(
            optimized_gradients, reference_gradients, strict=True
        )
    )
    retained_ratio = optimized_bytes / max(1, reference_bytes)
    return (
        (
            _criterion(
                "affine_scan_forward_error",
                forward_error,
                1e-11,
                "maximum",
                "absolute",
            ),
            _criterion(
                "affine_scan_gradient_error",
                gradient_error,
                1e-10,
                "maximum",
                "absolute",
            ),
            _criterion(
                "affine_scan_saved_tensor_byte_ratio",
                retained_ratio,
                0.75,
                "maximum",
                "ratio",
            ),
            _criterion(
                "affine_scan_saved_tensor_count",
                float(optimized_count),
                3.0,
                "maximum",
                "tensors",
            ),
        ),
        {
            "scan_reference_saved_tensors": reference_count,
            "scan_optimized_saved_tensors": optimized_count,
            "scan_reference_saved_bytes": reference_bytes,
            "scan_optimized_saved_bytes": optimized_bytes,
            "scan_reference_forward_seconds": reference_seconds,
            "scan_optimized_forward_seconds": optimized_seconds,
        },
    )


def _simplex_evidence() -> tuple[tuple[CarrierExecutionCriterion, ...], dict]:
    torch.manual_seed(1613)
    band = torch.randn(3, 29, 16, dtype=torch.float64)
    logits = torch.randn(3, 29, 5, dtype=torch.float64)
    scale = torch.tensor(0.11, dtype=torch.float64)
    branches = tuple(torch.randn_like(band) for _ in range(5))
    mask = torch.tensor(
        [
            [True] * 29,
            [True] * 23 + [False] * 6,
            [True] * 17 + [False] * 12,
        ]
    )
    cotangent = torch.randn_like(band)

    def leaves(values):
        return tuple(value.detach().clone().requires_grad_(True) for value in values)

    reference_inputs = leaves((band, logits, scale, *branches))
    optimized_inputs = leaves((band, logits, scale, *branches))
    rb, rl, rs, *rbranches = reference_inputs
    ob, ol, os, *obranches = optimized_inputs
    reference_weights = torch.softmax(rl, -1)
    reference_delta = sum(
        reference_weights[..., index : index + 1] * branch
        for index, branch in enumerate(rbranches)
    )
    reference = (rb + rs * reference_delta) * mask.unsqueeze(-1)
    optimized = fused_simplex_residual(
        ob, torch.softmax(ol, -1), os, mask, *obranches
    )
    reference_nodes = _graph_nodes(reference)
    optimized_nodes = _graph_nodes(optimized)
    reference_gradients = torch.autograd.grad(
        reference, reference_inputs, cotangent
    )
    optimized_gradients = torch.autograd.grad(
        optimized, optimized_inputs, cotangent
    )
    forward_error = float((optimized - reference).abs().max().detach())
    gradient_error = max(
        float((actual - expected).abs().max().detach())
        for actual, expected in zip(
            optimized_gradients, reference_gradients, strict=True
        )
    )
    return (
        (
            _criterion(
                "simplex_residual_forward_error",
                forward_error,
                1e-12,
                "maximum",
                "absolute",
            ),
            _criterion(
                "simplex_residual_gradient_error",
                gradient_error,
                1e-11,
                "maximum",
                "absolute",
            ),
            _criterion(
                "simplex_residual_autograd_node_ratio",
                optimized_nodes / max(1, reference_nodes),
                0.75,
                "maximum",
                "ratio",
            ),
        ),
        {
            "simplex_reference_autograd_nodes": reference_nodes,
            "simplex_optimized_autograd_nodes": optimized_nodes,
        },
    )


def _carrier_config(checkpointing: bool) -> MRRNConfig:
    return MRRNConfig(
        input_dim=6,
        model_dim=8,
        output_dim=17,
        layers=2,
        scales=3,
        heads=2,
        modes=2,
        mimo_rank=1,
        attention_window=4,
        attention_query_tile_size=4,
        retrieved_items=1,
        memory_capacity=4,
        mixer_expansion=1.5,
        width_growth_cap=1,
        mode_growth_cap=1,
        width_multiple=2,
        spectral_modes=2,
        spectral_basis_order=2,
        spectral_triads_per_mode=1,
        enable_global_head=False,
        relational_branch=True,
        relational_context_dim=8,
        activation_checkpointing=checkpointing,
    )


def _checkpoint_evidence() -> tuple[tuple[CarrierExecutionCriterion, ...], dict]:
    torch.manual_seed(1627)
    eager = MRRN(_carrier_config(False)).double().train()
    optimized = MRRN(_carrier_config(True)).double().train()
    optimized.load_state_dict(eager.state_dict())
    eager_input = torch.randn(
        2, 12, 6, dtype=torch.float64, requires_grad=True
    )
    optimized_input = eager_input.detach().clone().requires_grad_(True)
    mask = torch.tensor(
        [[True] * 12, [True] * 9 + [False] * 3]
    )
    reference = eager.prefill(
        eager_input, mask, project_output=False
    )
    actual = optimized.prefill_coarse_checkpointed(
        optimized_input, mask
    )
    reference.latent.square().mean().backward()
    actual.latent.square().mean().backward()
    forward_error = float(
        (actual.latent - reference.latent).abs().max().detach()
    )
    gradient_error = float(
        (optimized_input.grad - eager_input.grad).abs().max()
    )
    state_error = max(
        float((left.value - right.value).abs().max().detach())
        for left_block, right_block in zip(
            actual.state.blocks, reference.state.blocks, strict=True
        )
        for left, right in zip(
            left_block.resonators, right_block.resonators, strict=True
        )
    )
    receipt = optimized._last_composite_receipt
    return (
        (
            _criterion(
                "coarse_checkpoint_forward_error",
                forward_error,
                1e-10,
                "maximum",
                "absolute",
            ),
            _criterion(
                "coarse_checkpoint_input_gradient_error",
                gradient_error,
                1e-9,
                "maximum",
                "absolute",
            ),
            _criterion(
                "coarse_checkpoint_continuation_state_error",
                state_error,
                1e-10,
                "maximum",
                "absolute",
            ),
            _criterion(
                "coarse_checkpoint_receipt_present",
                float(receipt is not None),
                1.0,
                "minimum",
                "boolean",
            ),
        ),
        {
            "checkpoint_state_tensor_count": (
                0 if receipt is None else receipt.state_tensor_count
            ),
            "checkpoint_history_tensor_count": (
                0 if receipt is None else receipt.history_tensor_count
            ),
            "checkpoint_granularity": (
                "missing"
                if receipt is None
                else receipt.recomputation_granularity
            ),
        },
    )


def _document_evidence() -> tuple[tuple[CarrierExecutionCriterion, ...], dict]:
    tokenizer = ByteTextTokenizer()
    stream = PackedTokenStream(
        SequenceTextSource(
            ("alpha", "bravo", "cider", "delta", "ember", "fable")
        ),
        tokenizer,
    )
    batch = stream.next_batch(1, 32)
    planner = DocumentMajorBatchPlanner(
        tbptt_length=8,
        bucket_lengths=(2, 4, 8),
        token_budget=32,
        alignment=2,
        cognitive_stride=2,
    )
    plan = planner.plan(batch)
    logical_spans = sum(len(sequence.spans) for sequence in plan.sequences)
    invocation_ratio = plan.physical_invocations / logical_spans
    cost_ratio = (
        plan.cost_receipt.selected_estimated_cost
        / max(plan.cost_receipt.exact_signature_estimated_cost, 1e-30)
    )
    return (
        (
            _criterion(
                "document_target_bijection",
                float(plan.receipt.passed),
                1.0,
                "minimum",
                "boolean",
            ),
            _criterion(
                "document_physical_invocation_ratio",
                invocation_ratio,
                0.75,
                "maximum",
                "ratio",
            ),
            _criterion(
                "document_padding_efficiency",
                plan.padding_efficiency,
                0.60,
                "minimum",
                "ratio",
            ),
            _criterion(
                "document_cost_ratio_vs_exact_signature",
                cost_ratio,
                1.0,
                "maximum",
                "ratio",
            ),
        ),
        {
            "document_count": len(plan.sequences),
            "document_cohorts": len(plan.cohorts),
            "document_logical_spans": logical_spans,
            "document_physical_invocations": plan.physical_invocations,
            "document_physical_tokens": plan.physical_tokens,
            "document_valid_tokens": plan.valid_document_tokens,
            "document_target_digest": plan.receipt.planned_digest,
            "document_selected_estimated_cost": (
                plan.cost_receipt.selected_estimated_cost
            ),
            "document_exact_signature_estimated_cost": (
                plan.cost_receipt.exact_signature_estimated_cost
            ),
            "document_cost_ratio_vs_exact_signature": cost_ratio,
        },
    )


def _portable_device_evidence(
) -> tuple[tuple[CarrierExecutionCriterion, ...], dict]:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    criteria: list[CarrierExecutionCriterion] = []
    for device in devices:
        torch.manual_seed(1657)
        transition = torch.randn(
            2, 9, 2, 2, 2, device=device, requires_grad=True
        ) * 0.04
        transition[..., 0] = transition[..., 0] + 0.9
        drive = torch.randn_like(transition, requires_grad=True)
        initial = torch.randn(
            2, 2, 2, 2, device=device, requires_grad=True
        )
        scan = associative_affine_scan(
            transition, drive, initial, implementation="custom_adjoint"
        )
        band = torch.randn(2, 9, 8, device=device, requires_grad=True)
        logits = torch.randn(2, 9, 4, device=device, requires_grad=True)
        scale = torch.tensor(0.1, device=device, requires_grad=True)
        branches = tuple(
            torch.randn_like(band, requires_grad=True) for _ in range(4)
        )
        mask = torch.tensor(
            [[True] * 9, [True] * 6 + [False] * 3], device=device
        )
        residual = fused_simplex_residual(
            band, torch.softmax(logits, -1), scale, mask, *branches
        )
        (scan.square().mean() + residual.square().mean()).backward()
        finite = bool(
            torch.isfinite(scan).all()
            and torch.isfinite(residual).all()
            and all(
                value.grad is not None and torch.isfinite(value.grad).all()
                for value in (
                    drive, initial, band, logits, scale, *branches
                )
            )
        )
        criteria.append(
            _criterion(
                f"portable_{device}_forward_backward_finite",
                float(finite),
                1.0,
                "minimum",
                "boolean",
            )
        )
    return tuple(criteria), {
        "portable_devices_tested": ",".join(devices),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }


def run_carrier_execution_acceptance() -> CarrierExecutionAcceptanceReport:
    groups = (
        _scan_evidence(),
        _simplex_evidence(),
        _checkpoint_evidence(),
        _document_evidence(),
        _portable_device_evidence(),
    )
    criteria = tuple(item for group, _ in groups for item in group)
    telemetry: dict[str, float | int | str | bool] = {}
    for _, values in groups:
        telemetry.update(values)
    return CarrierExecutionAcceptanceReport(
        format_version=1,
        suite="mrcra-carrier-execution-optimization",
        device=str(telemetry["portable_devices_tested"]),
        dtype="float64-audit+float32-device-smoke",
        criteria=criteria,
        telemetry=telemetry,
        passed=all(item.passed for item in criteria),
        claim_boundary=(
            "Mechanism-level float64 CPU evidence for exact custom adjoints, "
            "coarse checkpoint continuity, and deterministic document batching, "
            "plus finite float32 forward/backward smoke tests on every locally "
            "available execution device. "
            "This does not claim a target GPU/MPS throughput multiplier."
        ),
    )
