from mrrn.vocabulary_router_acceptance import (
    run_vocabulary_router_acceptance,
)


def test_vocabulary_router_acceptance_proves_exactness_fallback_and_work_reduction():
    report = run_vocabulary_router_acceptance(
        seed=20260725,
        production_scale=False,
    )
    assert report.passed
    assert {criterion.name for criterion in report.criteria} == {
        "dense_reference_exactness",
        "fail_closed_fallback_and_staleness",
        "content_bound_serialization",
        "certified_work_reduction",
        "mlx_exact_authority",
        "portable_exact_training_authority",
    }
    assert all(experiment.exact_threshold_sets for experiment in report.experiments)
    assert all(experiment.avoided_fraction >= 0.50 for experiment in report.experiments)
    if report.mlx["available"]:
        assert report.mlx["exact_threshold_sets"]
        assert report.mlx["certificate_rate"] == 1
        if report.mlx["pytorch_mps_router_available"]:
            assert report.mlx["pytorch_mps_router_exact"]
            assert report.mlx["pytorch_mps_router_metadata_resident"]
    assert report.exact_training["native_tiled_exact"]
    if report.exact_training["native_tiled_mps_available"]:
        assert report.exact_training["native_tiled_mps_exact"]
    if report.exact_training["official_cce_available"]:
        assert report.exact_training["official_cce_exact"]
        if report.exact_training["official_cce_mps_available"]:
            assert report.exact_training["official_cce_mps_exact"]
    if report.exact_training["mlx_native_cce_available"]:
        assert report.exact_training["mlx_native_cce_exact"]
