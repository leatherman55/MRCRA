import math

import mpmath
import pytest
import torch

from mrrn import complex_ops as c


def random_pair(*shape, dtype=torch.float64):
    return torch.randn(*shape, 2, dtype=dtype)


def test_pair_validation_and_accessors():
    real, imag = torch.randn(3), torch.randn(3)
    z = c.pair(real, imag)
    torch.testing.assert_close(c.real(z), real)
    torch.testing.assert_close(c.imag(z), imag)
    torch.testing.assert_close(c.conjugate(z), c.pair(real, -imag))
    with pytest.raises(ValueError):
        c.pair(real, imag[:2])
    with pytest.raises(ValueError):
        c.pair(real, imag.to(torch.float64))
    with pytest.raises(ValueError):
        c.validate(torch.ones(3))
    with pytest.raises(TypeError):
        c.validate(torch.ones(3, 2, dtype=torch.int64))


def test_paired_real_arithmetic_matches_native_complex():
    a, b = random_pair(4, 3), random_pair(4, 3)
    an, bn = c.to_native(a), c.to_native(b)
    torch.testing.assert_close(c.to_native(c.add(a, b)), an + bn)
    torch.testing.assert_close(c.to_native(c.multiply(a, b)), an * bn)
    torch.testing.assert_close(c.to_native(c.divide(a, b)), an / bn)
    torch.testing.assert_close(c.to_native(c.exponential(a)), torch.exp(an))
    torch.testing.assert_close(c.abs_squared(a), an.abs().square())
    torch.testing.assert_close(c.magnitude(a), an.abs())


def test_scale_rotate_normalize_and_mod_silu_preserve_expected_geometry():
    z = random_pair(2, 5)
    factors = torch.rand(2, 5, dtype=z.dtype)
    torch.testing.assert_close(c.scale(z, 2.0), z * 2)
    torch.testing.assert_close(c.scale(z, factors), z * factors.unsqueeze(-1))
    angle = torch.full((2, 5), math.pi / 2, dtype=z.dtype)
    rotated = c.rotate(z, angle)
    torch.testing.assert_close(c.magnitude(rotated), c.magnitude(z))
    normalized = c.rms_normalize(z)
    torch.testing.assert_close(
        c.abs_squared(normalized).mean(-1), torch.ones(2, dtype=z.dtype), atol=2e-6, rtol=2e-6
    )
    activated = c.mod_silu(z, 0.2)
    cross = c.real(activated) * c.imag(z) - c.imag(activated) * c.real(z)
    torch.testing.assert_close(cross, torch.zeros_like(cross), atol=1e-10, rtol=0)


@pytest.mark.parametrize("magnitude", [0.0, 1e-8, 1e-4, 0.05, 1.0, 5.0])
def test_phi_functions_match_native_complex_reference(magnitude):
    q = torch.tensor([[[magnitude, -0.37 * magnitude]]], dtype=torch.float64)
    phi1, phi2 = c.phi_functions(q)
    qn = c.to_native(q)
    if magnitude == 0:
        expected1, expected2 = torch.ones_like(qn), torch.full_like(qn, 0.5)
    else:
        mpmath.mp.dps = 80
        high_q = mpmath.mpc(magnitude, -0.37 * magnitude)
        high_phi1 = mpmath.expm1(high_q) / high_q
        high_phi2 = (mpmath.exp(high_q) - 1 - high_q) / high_q**2
        expected1 = torch.tensor([[complex(high_phi1)]], dtype=torch.complex128)
        expected2 = torch.tensor([[complex(high_phi2)]], dtype=torch.complex128)
    torch.testing.assert_close(c.to_native(phi1), expected1, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(c.to_native(phi2), expected2, atol=1e-9, rtol=1e-7)


def test_gradient_is_finite_through_small_argument_branch():
    q = torch.tensor([[[1e-9, -2e-9]]], dtype=torch.float64, requires_grad=True)
    phi1, phi2 = c.phi_functions(q)
    (phi1.square().sum() + phi2.square().sum()).backward()
    assert torch.isfinite(q.grad).all()


def test_native_conversion_guards_and_rotation_dtype():
    native = torch.randn(2, dtype=torch.complex64)
    torch.testing.assert_close(c.to_native(c.from_native(native)), native)
    with pytest.raises(TypeError):
        c.from_native(torch.randn(2))
    with pytest.raises(TypeError):
        c.rotation(torch.ones(2, dtype=torch.int64))
    with pytest.raises(ValueError):
        c.rms_normalize(random_pair(2, 3), dim=-1)
