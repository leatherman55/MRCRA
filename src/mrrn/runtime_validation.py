"""Scoped validation policy for trusted, repeatedly constructed tensor state.

Public inputs, restored checkpoints, and final span state are always validated.
The integrated training hot loop may suppress redundant dataclass validation
while it constructs intermediate states, then validate the resulting state tree
once at the span boundary.  This removes device synchronizations and duplicate
small tensor kernels without changing any state transition.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from typing import Iterator


_VALIDATION_ENABLED: ContextVar[bool] = ContextVar(
    "mrrn_runtime_validation_enabled", default=True,
)


def runtime_validation_enabled() -> bool:
    """Return whether tensor-state constructors must run invariant checks."""

    return _VALIDATION_ENABLED.get()


@contextmanager
def defer_runtime_validation() -> Iterator[None]:
    """Defer repeated internal checks until an explicit boundary validation."""

    token = _VALIDATION_ENABLED.set(False)
    try:
        yield
    finally:
        _VALIDATION_ENABLED.reset(token)


def validate_dataclass_tree(value: object) -> None:
    """Recursively run declared dataclass invariants on a completed state tree."""

    visited: set[int] = set()

    def visit(item: object) -> None:
        identity = id(item)
        if identity in visited or not is_dataclass(item):
            return
        visited.add(identity)
        validator = type(item).__dict__.get("__post_init__")
        if validator is not None:
            validator(item)
        for field in fields(item):
            child = getattr(item, field.name)
            if is_dataclass(child):
                visit(child)
            elif isinstance(child, (tuple, list)):
                for member in child:
                    if is_dataclass(member):
                        visit(member)

    visit(value)
