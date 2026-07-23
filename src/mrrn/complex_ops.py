"""Paired-real complex operations used by all resonance paths.

A complex tensor has shape ``(..., 2)`` with real and imaginary values in the
last dimension. Public storage stays paired-real for portable checkpoints.
Metal hot paths use explicit paired-real arithmetic for fusion; CPU hot paths
use optimized native-complex kernels at profitable operation boundaries.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def validate(z: Tensor) -> Tensor:
    if z.ndim == 0 or z.shape[-1] != 2:
        raise ValueError(f"paired-real complex tensors require final dimension 2, got {z.shape}")
    if not z.is_floating_point():
        raise TypeError("paired-real complex tensors must use a floating dtype")
    return z


def pair(real: Tensor, imag: Tensor) -> Tensor:
    if real.shape != imag.shape:
        raise ValueError(f"real/imaginary shapes differ: {real.shape} != {imag.shape}")
    if real.dtype != imag.dtype or real.device != imag.device:
        raise ValueError("real and imaginary tensors must share dtype and device")
    return torch.stack((real, imag), dim=-1)


def real(z: Tensor) -> Tensor:
    return validate(z)[..., 0]


def imag(z: Tensor) -> Tensor:
    return validate(z)[..., 1]


def conjugate(z: Tensor) -> Tensor:
    z = validate(z)
    if z.device.type == "cpu":
        return torch.view_as_real(
            torch.view_as_complex(z).conj().resolve_conj()
        )
    return torch.stack((z[..., 0], -z[..., 1]), dim=-1)


def add(a: Tensor, b: Tensor) -> Tensor:
    validate(a)
    validate(b)
    return a + b


def multiply(a: Tensor, b: Tensor) -> Tensor:
    a, b = torch.broadcast_tensors(validate(a), validate(b))
    if a.device.type == "cpu":
        return torch.view_as_real(torch.view_as_complex(a) * torch.view_as_complex(b))
    real = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
    imag = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    return torch.stack((real, imag), dim=-1)


def scale(z: Tensor, factor: Tensor | float) -> Tensor:
    z = validate(z)
    factor_tensor = torch.as_tensor(factor, dtype=z.dtype, device=z.device)
    return z * factor_tensor.unsqueeze(-1) if factor_tensor.ndim == z.ndim - 1 else z * factor_tensor


def abs_squared(z: Tensor) -> Tensor:
    z = validate(z)
    if z.device.type == "cpu":
        return torch.view_as_complex(z).abs().square()
    return z[..., 0].square() + z[..., 1].square()


def magnitude(z: Tensor, eps: float = 0.0) -> Tensor:
    return (abs_squared(z) + eps).sqrt()


def divide(a: Tensor, b: Tensor, eps: float = 0.0) -> Tensor:
    a, b = torch.broadcast_tensors(validate(a), validate(b))
    denominator = abs_squared(b) + eps
    return scale(multiply(a, conjugate(b)), denominator.reciprocal())


def exponential(z: Tensor) -> Tensor:
    z = validate(z)
    if z.device.type == "cpu":
        return torch.view_as_real(torch.exp(torch.view_as_complex(z)))
    amplitude = torch.exp(z[..., 0])
    return torch.stack(
        (amplitude * torch.cos(z[..., 1]), amplitude * torch.sin(z[..., 1])),
        dim=-1,
    )


def rotation(angle: Tensor) -> Tensor:
    if not angle.is_floating_point():
        raise TypeError("angle must use a floating dtype")
    if angle.device.type == "cpu":
        return torch.view_as_real(torch.polar(torch.ones_like(angle), angle))
    return torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)


def rotate(z: Tensor, angle: Tensor) -> Tensor:
    z = validate(z)
    if z.device.type == "cpu":
        native = torch.view_as_complex(z)
        phase = torch.polar(torch.ones_like(angle), angle)
        return torch.view_as_real(native * phase)
    return multiply(z, rotation(angle))


def rms_normalize(z: Tensor, dim: int = -2, eps: float = 1e-6) -> Tensor:
    z = validate(z)
    complex_axis = dim % z.ndim
    if complex_axis == z.ndim - 1:
        raise ValueError("cannot normalize across the paired real/imaginary axis")
    rms = (abs_squared(z).mean(dim=complex_axis, keepdim=True) + eps).sqrt()
    return scale(z, rms.reciprocal())


def mod_silu(z: Tensor, bias: Tensor | float = 0.0, eps: float = 1e-6) -> Tensor:
    z = validate(z)
    amplitude = magnitude(z)
    bias_tensor = torch.as_tensor(bias, dtype=z.dtype, device=z.device)
    gain = F.silu(amplitude + bias_tensor) / (amplitude + eps)
    return scale(z, gain)


def _constant_like(z: Tensor, value: float) -> Tensor:
    return pair(torch.full_like(z[..., 0], value), torch.zeros_like(z[..., 1]))


def phi_functions(q: Tensor, threshold: float | None = None) -> tuple[Tensor, Tensor]:
    """Return stable paired-real phi_1 and phi_2 for complex ``q``."""

    q = validate(q)
    if threshold is None:
        threshold = 3e-2 if q.dtype in (torch.float16, torch.bfloat16) else 1e-3
    one = _constant_like(q, 1.0)
    half = _constant_like(q, 0.5)
    q2 = multiply(q, q)
    q3 = multiply(q2, q)
    exp_q = exponential(q)
    general_phi1 = divide(exp_q - one, q, eps=torch.finfo(q.dtype).tiny)
    general_phi2 = divide(exp_q - one - q, q2, eps=torch.finfo(q.dtype).tiny)
    series_phi1 = one + scale(q, 0.5) + scale(q2, 1.0 / 6.0) + scale(q3, 1.0 / 24.0)
    series_phi2 = half + scale(q, 1.0 / 6.0) + scale(q2, 1.0 / 24.0) + scale(q3, 1.0 / 120.0)
    small = (abs_squared(q) < threshold**2).unsqueeze(-1)
    return torch.where(small, series_phi1, general_phi1), torch.where(
        small, series_phi2, general_phi2
    )


def to_native(z: Tensor) -> Tensor:
    z = validate(z).contiguous()
    return torch.view_as_complex(z)


def from_native(z: Tensor) -> Tensor:
    if not z.is_complex():
        raise TypeError("expected a native complex tensor")
    return torch.view_as_real(z)
