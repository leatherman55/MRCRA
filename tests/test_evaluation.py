import json

import pytest
import torch

from mrrn.config import MRRNConfig
from mrrn.evaluation import (
    ABLATION_MATRIX,
    apply_ablation,
    CausalTransformerBaseline,
    LongConvolutionBaseline,
    RealSelectiveSSMBaseline,
    benchmark,
    benchmark_suite,
    environment_report,
    local_path_ablation,
    make_reference_baselines,
    matched_width,
    parameter_statistics,
    save_benchmark_report,
)
from mrrn.model import MRRN


def test_all_reference_baseline_decode_paths_match_their_batch_paths():
    torch.manual_seed(205)
    baselines = (
        CausalTransformerBaseline(3, 8, 4, 2, 2),
        RealSelectiveSSMBaseline(3, 8, 4, 2),
        LongConvolutionBaseline(3, 8, 4, 2, kernel_size=5),
    )
    x = torch.randn(2, 9, 3)
    for model in baselines:
        model.eval()
        with torch.no_grad():
            expected = model(x)
            state = model.initial_decode_state(2)
            actual = []
            for position in range(x.shape[1]):
                output, state = model.step(x[:, position], state)
                actual.append(output.unsqueeze(1))
        torch.testing.assert_close(torch.cat(actual, 1), expected, atol=2e-6, rtol=2e-6)


def tiny_config():
    return MRRNConfig(
        input_dim=3, model_dim=4, output_dim=2, layers=1, scales=2, heads=1,
        modes=2, mimo_rank=1, attention_window=2, retrieved_items=1,
        memory_capacity=2, mixer_expansion=1, width_growth_cap=1,
        mode_growth_cap=1, width_multiple=1,
    )


def test_end_to_end_benchmark_measures_training_prefill_decode_and_serializes_environment(tmp_path):
    torch.manual_seed(211)
    config = tiny_config()
    models = {"mrrn": MRRN(config), **make_reference_baselines(config)}
    x = torch.randn(1, 4, 3)
    results = benchmark_suite(models, x, repeats=1, warmup=0)
    assert len(results) == 12
    assert {result.phase for result in results} == {"training", "prefill", "decode"}
    assert all(result.parameters > 0 and result.timing.mean_seconds > 0 for result in results)
    path = tmp_path / "benchmark.json"
    save_benchmark_report(path, results, seed=211, notes={"scope": "unit smoke"})
    payload = json.loads(path.read_text())
    assert payload["seed"] == 211 and payload["environment"]["torch"]
    assert environment_report()["device"]


def test_parameter_matching_and_local_ablation_are_measured_not_assumed():
    target = parameter_statistics(RealSelectiveSSMBaseline(2, 16, 2, 1))[0]
    width = matched_width(lambda value: RealSelectiveSSMBaseline(2, value, 2, 1), target, maximum=32)
    assert width == 16
    model = MRRN(tiny_config())
    local_path_ablation(model)
    for block in model.blocks:
        assert (block.exchange.fine_gain == 0).all()
        assert all(gate.bias.argmax() == 1 for gate in block.branch_gates)
    assert len(ABLATION_MATRIX) == 13 and "euler_vs_exponential_trapezoid" in ABLATION_MATRIX


@pytest.mark.parametrize(
    "variant",
    [
        "local_mixer_only", "no_cross_scale", "fine_to_coarse_only",
        "coarse_to_fine_only", "no_attention", "fixed_poles",
        "fixed_haar", "magnitude_only_keys",
    ],
)
def test_executable_ablation_variants_are_isolated_and_preserve_output_contract(variant):
    full = MRRN(tiny_config())
    ablated = apply_ablation(full, variant)
    assert ablated is not full
    assert ablated(torch.randn(1, 7, 3)).prediction.shape == (1, 7, 2)
    if variant == "fixed_haar":
        assert not any(parameter.requires_grad for parameter in ablated.analysis.parameters())


def test_unknown_ablation_fails_closed():
    with pytest.raises(ValueError):
        apply_ablation(MRRN(tiny_config()), "pretend_ablation")


def test_benchmark_and_baseline_contracts_fail_closed():
    with pytest.raises(ValueError):
        CausalTransformerBaseline(2, 7, 2, 2, 1)
    with pytest.raises(ValueError):
        RealSelectiveSSMBaseline(2, 0, 2, 1)
    with pytest.raises(ValueError):
        LongConvolutionBaseline(2, 3, 2, 1, 0)
    with pytest.raises(ValueError):
        benchmark("bad", RealSelectiveSSMBaseline(2, 3, 2, 1), torch.randn(1, 2, 2), phase="bad")
    with pytest.raises(ValueError):
        matched_width(lambda value: torch.nn.Linear(value, 1), 0)
