"""Compatibility exports for the format-16 CSTM scheduling authority.

The implementation moved to :mod:`mrrn.cstm_schedule` so the production module
name reflects that its authority includes duty scheduling, obligation
selection, row sampling, and persistent coverage. Existing import paths remain
valid for early format-16 callers and external experiments.
"""

from .cstm_schedule import (
    CSTM_COVERAGE_SCHEMA_VERSION,
    CSTM_SAMPLING_SCHEMA_VERSION,
    CSTMCoverageState,
    CSTMObligation,
    CSTMRowSamplingDecision,
    CSTMSamplingDecision,
    deterministic_cstm_rows,
    deterministic_cstm_sample,
)

__all__ = [
    "CSTM_COVERAGE_SCHEMA_VERSION",
    "CSTM_SAMPLING_SCHEMA_VERSION",
    "CSTMCoverageState",
    "CSTMObligation",
    "CSTMRowSamplingDecision",
    "CSTMSamplingDecision",
    "deterministic_cstm_rows",
    "deterministic_cstm_sample",
]
