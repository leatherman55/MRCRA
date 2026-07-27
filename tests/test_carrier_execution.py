"""Production acceptance tests for coarse carrier execution composites."""

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gc
import multiprocessing
import pickle
import tracemalloc

import pytest
import torch

import mrrn.model as model_module
from mrrn.carrier_execution import (
    CarrierCompilationRegistry,
    CompiledCarrierShapeKey,
    TensorTreeSpec,
    flatten_tensor_tree,
    fused_simplex_residual,
    resolve_carrier_execution_policy,
    unflatten_tensor_tree,
)
from mrrn.config import MRRNConfig
from mrrn.model import MRRN
from mrrn.resonance import associative_affine_scan


AVAILABLE_EXECUTION_DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    AVAILABLE_EXECUTION_DEVICES.append("mps")
if torch.cuda.is_available():
    AVAILABLE_EXECUTION_DEVICES.append("cuda")


def carrier_config(*, checkpointing: bool) -> MRRNConfig:
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


def test_tensor_tree_codec_roundtrips_all_authorized_containers_without_copying():
    first = torch.randn(2, 3, requires_grad=True)
    second = torch.arange(4)
    authority = {
        "state": [first, None, (7, second)],
        "identity": {"kind": "carrier", "enabled": True, "scale": 0.5},
    }
    tensors, spec = flatten_tensor_tree(authority)
    restored = unflatten_tensor_tree(tensors, spec)
    assert isinstance(spec, TensorTreeSpec)
    assert len(spec.digest) == 64
    assert restored["state"][0] is first
    assert restored["state"][2][1] is second
    assert restored["identity"] == authority["identity"]
    with pytest.raises(ValueError, match="expected"):
        unflatten_tensor_tree(tensors[:-1], spec)
    with pytest.raises(TypeError, match="support"):
        flatten_tensor_tree({"unsupported": object()})


def compiled_key(*, length: int = 128) -> CompiledCarrierShapeKey:
    return CompiledCarrierShapeKey(
        model_semantic_digest="a" * 64,
        device="cpu",
        dtype="torch.float32",
        batch=2,
        padded_length=length,
        scale=1,
        activation_policy="retain",
        torch_version=str(torch.__version__),
    )


def _spawn_registry_roundtrip(
    state: dict[str, object],
    result_queue,
) -> None:
    """Spawn-safe worker proving registry locks are process-local."""

    registry = CarrierCompilationRegistry.from_state_dict(state)
    registry.register(compiled_key(length=256))
    registry.record_fallback("SpawnProbe")
    result_queue.put(registry.state_dict())


def test_compilation_registry_is_bounded_thread_safe_and_digest_bound():
    registry = CarrierCompilationRegistry(maximum_shapes=2)
    key = compiled_key()
    with ThreadPoolExecutor(max_workers=8) as executor:
        created = list(executor.map(lambda _: registry.register(key), range(32)))
    assert sum(created) == 1
    registry.record_first_execution(key, 0.25)
    assert registry.shape_count == 1
    assert registry.compile_seconds == 0.25
    assert registry.register(compiled_key(length=256))
    with pytest.raises(RuntimeError, match="exceeded"):
        registry.register(compiled_key(length=512))
    restored = CarrierCompilationRegistry.from_state_dict(
        registry.state_dict()
    )
    assert restored.state_dict() == registry.state_dict()
    copied = deepcopy(registry)
    unpickled = pickle.loads(pickle.dumps(registry))
    assert copied.state_dict() == registry.state_dict()
    assert unpickled.state_dict() == registry.state_dict()
    assert copied._lock is not registry._lock
    assert unpickled._lock is not registry._lock


def test_compilation_registry_roundtrips_through_multiprocessing_spawn():
    """A compiled-plan receipt must be portable without sharing a mutex."""

    registry = CarrierCompilationRegistry(maximum_shapes=4)
    registry.register(compiled_key())
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_spawn_registry_roundtrip,
        args=(registry.state_dict(), queue),
    )
    process.start()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail("spawned registry worker did not terminate")
    assert process.exitcode == 0
    restored = CarrierCompilationRegistry.from_state_dict(
        queue.get(timeout=5)
    )
    queue.close()
    queue.join_thread()
    assert restored.shape_count == 2
    assert restored.fallback_count == 1
    assert restored.fallback_reasons == {"SpawnProbe": 1}
    assert registry.shape_count == 1
    assert registry.fallback_count == 0


def test_compilation_registry_corruption_fails_closed():
    registry = CarrierCompilationRegistry()
    registry.register(compiled_key())
    state = registry.state_dict()
    state["entries"][0]["digest"] = "b" * 64
    with pytest.raises(ValueError, match="registry"):
        CarrierCompilationRegistry.from_state_dict(state)


def test_compilation_registry_fallback_reason_corruption_fails_closed():
    registry = CarrierCompilationRegistry()
    registry.register(compiled_key())
    registry.record_fallback("GraphBreak")
    state = registry.state_dict()
    state["fallback_reasons"]["GraphBreak"] = 2
    with pytest.raises(ValueError, match="registry"):
        CarrierCompilationRegistry.from_state_dict(state)


@pytest.mark.parametrize(
    ("device", "requested", "compiled"),
    [
        ("cpu", None, False),
        ("mps", None, False),
        ("cuda", None, True),
        ("cpu", True, True),
        ("cuda", False, False),
    ],
)
def test_backend_policy_keeps_portable_custom_authority_on_every_device(
    device, requested, compiled,
):
    policy = resolve_carrier_execution_policy(
        device_type=device,
        compile_tensor_cores=requested,
        integrated=True,
        activation_checkpointing=True,
    )
    assert policy.compiler_enabled is compiled
    assert policy.compiler_backend == (
        "none"
        if not compiled
        else "inductor"
        if device == "cuda"
        else "aot_eager"
    )
    assert policy.affine_scan == "custom_paired_real_adjoint"
    assert policy.simplex_residual == "custom_simplex_residual_adjoint"
    assert policy.checkpoint_granularity == "whole_carrier_span"
    assert "portable" in policy.fallback
    with pytest.raises(ValueError, match="unsupported"):
        resolve_carrier_execution_policy(
            device_type="tpu",
            compile_tensor_cores=None,
            integrated=True,
            activation_checkpointing=True,
        )


def test_carrier_compiler_backend_contract_fails_closed():
    model = MRRN(carrier_config(checkpointing=False))
    with pytest.raises(ValueError, match="compiler backend"):
        model.enable_compiled_tensor_cores(backend="unknown")


@pytest.mark.parametrize("device", AVAILABLE_EXECUTION_DEVICES)
def test_portable_custom_composites_execute_finite_forward_and_backward_on_device(
    device,
):
    torch.manual_seed(1193)
    transition = torch.randn(
        2, 9, 2, 3, 2, device=device, requires_grad=True
    ) * 0.05
    transition[..., 0] = transition[..., 0] + 0.9
    drive = torch.randn_like(transition, requires_grad=True)
    initial = torch.randn(
        2, 2, 3, 2, device=device, requires_grad=True
    )
    scan = associative_affine_scan(
        transition, drive, initial, implementation="custom_adjoint"
    )
    band = torch.randn(2, 9, 7, device=device, requires_grad=True)
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
    assert torch.isfinite(scan).all()
    assert torch.isfinite(residual).all()
    for value in (drive, initial, band, logits, scale, *branches):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


@pytest.mark.parametrize("branch_count", [4, 5])
def test_fused_simplex_residual_matches_reference_forward_and_full_adjoint(
    branch_count,
):
    torch.manual_seed(1181 + branch_count)
    band = torch.randn(2, 7, 5, dtype=torch.float64)
    logits = torch.randn(2, 7, branch_count, dtype=torch.float64)
    scale = torch.tensor(0.17, dtype=torch.float64)
    branches = tuple(
        torch.randn_like(band) for _ in range(branch_count)
    )
    mask = torch.tensor(
        [[True] * 7, [True] * 4 + [False] * 3]
    )
    cotangent = torch.randn_like(band)

    def leaves(values):
        return tuple(value.detach().clone().requires_grad_(True) for value in values)

    reference_inputs = leaves((band, logits, scale, *branches))
    fused_inputs = leaves((band, logits, scale, *branches))
    reference_band, reference_logits, reference_scale, *reference_branches = (
        reference_inputs
    )
    fused_band, fused_logits, fused_scale, *fused_branches = fused_inputs
    reference_weights = torch.softmax(reference_logits, -1)
    delta = sum(
        reference_weights[..., index : index + 1] * branch
        for index, branch in enumerate(reference_branches)
    )
    reference = (
        reference_band + reference_scale * delta
    ) * mask.unsqueeze(-1)
    actual_weights = torch.softmax(fused_logits, -1)
    actual = fused_simplex_residual(
        fused_band,
        actual_weights,
        fused_scale,
        mask,
        *fused_branches,
    )

    def graph_nodes(value):
        pending = [value.grad_fn]
        seen = set()
        while pending:
            node = pending.pop()
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            pending.extend(next_node for next_node, _ in node.next_functions)
        return len(seen)

    assert graph_nodes(actual) < graph_nodes(reference)
    reference_gradients = torch.autograd.grad(
        reference, reference_inputs, cotangent
    )
    actual_gradients = torch.autograd.grad(
        actual, fused_inputs, cotangent
    )
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    for actual_gradient, reference_gradient in zip(
        actual_gradients, reference_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient,
            reference_gradient,
            atol=2e-12,
            rtol=2e-12,
        )
    assert torch.count_nonzero(actual[~mask]) == 0
    assert "SimplexResidualAdjoint" in type(actual.grad_fn).__name__


def _carrier_loss(output) -> torch.Tensor:
    value = output.latent.square().mean()
    for block in output.state.blocks:
        for resonator in block.resonators:
            value = value + 1e-3 * resonator.value.square().mean()
    for history in output.band_histories:
        if history is not None:
            value = value + 1e-3 * history.band.data.square().mean()
    return value


def test_whole_span_checkpoint_matches_eager_outputs_states_histories_and_gradients(
    monkeypatch,
):
    """One coarse recomputation boundary must preserve every live authority."""

    torch.manual_seed(1201)
    eager = MRRN(carrier_config(checkpointing=False)).double().train()
    optimized = MRRN(carrier_config(checkpointing=True)).double().train()
    optimized.load_state_dict(eager.state_dict())
    eager_input = torch.randn(
        2, 12, 6, dtype=torch.float64, requires_grad=True
    )
    optimized_input = eager_input.detach().clone().requires_grad_(True)
    mask = torch.tensor(
        [
            [True] * 12,
            [True] * 9 + [False] * 3,
        ]
    )
    relational = torch.randn(2, 8, dtype=torch.float64)

    checkpoint_calls = 0
    original_checkpoint = model_module.checkpoint

    def counted_checkpoint(function, *arguments, **options):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        return original_checkpoint(function, *arguments, **options)

    monkeypatch.setattr(model_module, "checkpoint", counted_checkpoint)
    expected = eager.prefill(
        eager_input,
        mask,
        relational_context=relational,
        project_output=False,
    )
    actual = optimized.prefill_coarse_checkpointed(
        optimized_input,
        mask,
        relational_context=relational,
    )
    _carrier_loss(expected).backward()
    _carrier_loss(actual).backward()

    assert checkpoint_calls == 1
    assert optimized._last_composite_receipt is not None
    assert (
        optimized._last_composite_receipt.recomputation_granularity
        == "whole_carrier_span"
    )
    torch.testing.assert_close(actual.latent, expected.latent, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(
        optimized_input.grad, eager_input.grad, atol=2e-10, rtol=2e-10
    )
    for actual_history, expected_history in zip(
        actual.band_histories, expected.band_histories, strict=True
    ):
        assert actual_history is not None and expected_history is not None
        torch.testing.assert_close(
            actual_history.band.data,
            expected_history.band.data,
            atol=2e-11,
            rtol=2e-11,
        )
        assert torch.equal(
            actual_history.end_positions, expected_history.end_positions
        )
    for actual_block, expected_block in zip(
        actual.state.blocks, expected.state.blocks, strict=True
    ):
        for actual_resonator, expected_resonator in zip(
            actual_block.resonators,
            expected_block.resonators,
            strict=True,
        ):
            torch.testing.assert_close(
                actual_resonator.value,
                expected_resonator.value,
                atol=2e-11,
                rtol=2e-11,
            )
    eager_parameters = dict(eager.named_parameters())
    for name, parameter in optimized.named_parameters():
        reference = eager_parameters[name]
        if reference.grad is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(
                parameter.grad,
                reference.grad,
                atol=3e-9,
                rtol=3e-9,
                msg=name,
            )


def test_retain_selective_and_whole_span_match_float64_complete_adjoint():
    """Every activation policy must be one mathematical carrier."""

    torch.manual_seed(1209)
    base = MRRN(carrier_config(checkpointing=False)).double().train()
    initial = deepcopy(base.state_dict())
    values = torch.randn(2, 12, 6, dtype=torch.float64)
    mask = torch.tensor(
        [[True] * 12, [True] * 8 + [False] * 4]
    )
    relational = torch.randn(2, 8, dtype=torch.float64)
    observations = {}
    for policy in ("retain", "selective", "whole_span"):
        model = MRRN(carrier_config(checkpointing=False)).double().train()
        model.load_state_dict(initial)
        model.configure_activation_execution(
            policy,
            selective_scales=(0, 2) if policy == "selective" else (),
        )
        local = values.detach().clone().requires_grad_(True)
        output = (
            model.prefill_coarse_checkpointed(
                local,
                mask,
                relational_context=relational,
            )
            if policy == "whole_span"
            else model.prefill(
                local,
                mask,
                relational_context=relational,
                project_output=False,
            )
        )
        _carrier_loss(output).backward()
        observations[policy] = (
            output,
            local.grad.detach().clone(),
            {
                name: (
                    None
                    if parameter.grad is None
                    else parameter.grad.detach().clone()
                )
                for name, parameter in model.named_parameters()
            },
        )

    reference_output, reference_input_gradient, reference_parameters = (
        observations["retain"]
    )
    for policy in ("selective", "whole_span"):
        output, input_gradient, parameters = observations[policy]
        torch.testing.assert_close(
            output.latent,
            reference_output.latent,
            atol=2e-10,
            rtol=2e-10,
        )
        torch.testing.assert_close(
            input_gradient,
            reference_input_gradient,
            atol=5e-9,
            rtol=5e-9,
        )
        for actual_block, expected_block in zip(
            output.state.blocks,
            reference_output.state.blocks,
            strict=True,
        ):
            for actual, expected in zip(
                actual_block.resonators,
                expected_block.resonators,
                strict=True,
            ):
                torch.testing.assert_close(
                    actual.value,
                    expected.value,
                    atol=2e-10,
                    rtol=2e-10,
                )
        for name, expected in reference_parameters.items():
            actual = parameters[name]
            assert (actual is None) == (expected is None), name
            if actual is not None:
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=5e-9,
                    rtol=5e-9,
                    msg=name,
                )


def test_coarse_checkpoint_continuation_is_exact_across_two_static_spans():
    torch.manual_seed(1217)
    eager = MRRN(carrier_config(checkpointing=False)).double().eval()
    optimized = MRRN(carrier_config(checkpointing=True)).double().eval()
    optimized.load_state_dict(eager.state_dict())
    values = torch.randn(2, 16, 6, dtype=torch.float64)
    mask = torch.tensor(
        [[True] * 16, [True] * 13 + [False] * 3]
    )
    eager_state = optimized_state = None
    eager_latents, optimized_latents = [], []
    with torch.no_grad():
        for start in (0, 8):
            expected = eager.prefill(
                values[:, start : start + 8],
                mask[:, start : start + 8],
                state=eager_state,
                project_output=False,
            )
            actual = optimized.prefill_coarse_checkpointed(
                values[:, start : start + 8],
                mask[:, start : start + 8],
                state=optimized_state,
            )
            eager_state, optimized_state = expected.state, actual.state
            eager_latents.append(expected.latent)
            optimized_latents.append(actual.latent)
    torch.testing.assert_close(
        torch.cat(optimized_latents, 1),
        torch.cat(eager_latents, 1),
        atol=2e-11,
        rtol=2e-11,
    )
    assert optimized_state.position == eager_state.position == 16


def test_compiler_graph_break_falls_back_to_exact_portable_carrier_and_is_visible():
    """A failed full graph must preserve semantics and publish its fallback."""

    class DeliberateGraphBreak:
        def __call__(self, *unused):
            raise RuntimeError("deliberate full-graph rejection")

    torch.manual_seed(1229)
    reference = MRRN(carrier_config(checkpointing=False)).eval()
    candidate = MRRN(carrier_config(checkpointing=False)).eval()
    candidate.load_state_dict(reference.state_dict())
    failing = DeliberateGraphBreak()
    for block in candidate.blocks:
        block._compiled_chunk_cores = {
            scale: failing for scale in range(candidate.config.scales)
        }
    values = torch.randn(2, 12, 6)
    mask = torch.tensor(
        [[True] * 12, [True] * 7 + [False] * 5]
    )
    with torch.no_grad():
        expected = reference.prefill(
            values, mask, project_output=False
        )
        actual = candidate.prefill(values, mask, project_output=False)
    torch.testing.assert_close(
        actual.latent, expected.latent, atol=3e-6, rtol=3e-5
    )
    receipt = candidate.compilation_receipt()
    assert receipt["fallback_count"] > 0
    assert receipt["graph_break_count"] == receipt["fallback_count"]
    assert receipt["fallback_reasons"] == {
        "RuntimeError": int(receipt["fallback_count"])
    }
    assert receipt["compiled_shape_count"] <= (
        candidate.config.layers * candidate.config.scales
    )


def test_repeated_portable_forward_has_bounded_python_memory_and_cache_growth():
    """Steady-shape inference must not retain per-forward Python authority."""

    torch.manual_seed(1237)
    model = MRRN(carrier_config(checkpointing=False)).eval()
    values = torch.randn(1, 16, 6)
    mask = torch.ones(1, 16, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(4):
            output = model.prefill(values, mask, project_output=False)
    del output
    gc.collect()
    tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    with torch.no_grad():
        for _ in range(32):
            output = model.prefill(values, mask, project_output=False)
            assert torch.isfinite(output.latent).all()
    del output
    gc.collect()
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert retained - baseline < 2 << 20
    assert peak - baseline < 16 << 20
    assert model.compilation_receipt()["compiled_shape_count"] == 0
