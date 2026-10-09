"""Numerical/gradient parity for the full-scale deterministic bilinear backend."""
import pytest
import torch
import torch.nn.functional as F

from src.models.bilinear_sampling import (
    deterministic_bilinear_sample, deterministic_bilinear_resize,
)


@pytest.mark.parametrize('padding', ['zeros', 'border'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('hw', [(1, 1), (3, 7), (32, 32)])
def test_fixed_grid_forward_and_input_gradient_match_reference(padding, dtype, hw):
    generator = torch.Generator().manual_seed(13)
    x = torch.randn(2, 3, *hw, dtype=dtype, generator=generator).requires_grad_()
    grid = torch.rand(2, 5, 11, 2, dtype=dtype, generator=generator) * 3 - 1.5
    # Pixel edge and corner conventions matter for both native and feature grids.
    grid[:, 0, :4] = torch.tensor([[-1., -1.], [1., 1.], [-1., 1.], [1., -1.]], dtype=dtype)
    expected = F.grid_sample(x, grid, mode='bilinear', padding_mode=padding, align_corners=False)
    actual = deterministic_bilinear_sample(x, grid, padding_mode=padding)
    upstream = torch.randn(actual.shape, dtype=dtype, generator=generator)
    expected_grad, = torch.autograd.grad(expected, x, upstream)
    actual_grad, = torch.autograd.grad(actual, x, upstream)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_grad, expected_grad, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('source,target', [((1, 1), (13, 17)), ((3, 7), (1, 1)),
                                          ((5, 9), (13, 17)), ((9, 7), (3, 4)),
                                          ((32, 32), (512, 512))])
def test_resize_forward_and_gradient_match_reference(dtype, source, target):
    generator = torch.Generator().manual_seed(31)
    x = torch.randn(2, 1, *source, dtype=dtype, generator=generator).requires_grad_()
    expected = F.interpolate(x, size=target, mode='bilinear', align_corners=False)
    actual = deterministic_bilinear_resize(x, target)
    upstream = torch.randn(actual.shape, dtype=dtype, generator=generator)
    expected_grad, = torch.autograd.grad(expected, x, upstream)
    actual_grad, = torch.autograd.grad(actual, x, upstream)
    tolerance = 2e-4 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_grad, expected_grad, atol=tolerance, rtol=tolerance)


def test_learnable_geometry_is_rejected():
    with pytest.raises(ValueError, match='fixed geometry'):
        deterministic_bilinear_sample(torch.ones(1, 1, 2, 2), torch.zeros(1, 2, 2, 2, requires_grad=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA for strict deterministic backward')
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_cuda_strict_deterministic_alignment_resize_backward(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('GPU does not support bfloat16')
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        values = torch.randn(2, 9, 32, 32, device='cuda', dtype=dtype)
        grid = torch.rand(2, 32, 32, 2, device='cuda') * 2 - 1
        results = []
        for _ in range(2):
            x = values.clone().requires_grad_()
            aligned = deterministic_bilinear_sample(x, grid)
            resized = deterministic_bilinear_resize(aligned, (512, 512))
            resized.float().square().mean().backward()
            results.append((resized.detach().clone(), x.grad.clone()))
        assert all(torch.equal(a, b) for a, b in zip(*results))
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
