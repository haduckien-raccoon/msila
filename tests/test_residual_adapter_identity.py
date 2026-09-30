import torch

from models.residual_adapter import ResidualAdapter2d


def test_adapter_preserves_shape():
    adapter = ResidualAdapter2d(
        in_channels=384,
        bottleneck_channels=96,
    )

    x = torch.randn(2, 384, 32, 32)

    y = adapter(x)

    assert y.shape == x.shape


def test_gamma_zero_is_exact_identity():
    adapter = ResidualAdapter2d(
        in_channels=384,
        bottleneck_channels=96,
        gamma_init=0.0,
    )

    x = torch.randn(2, 384, 16, 16)

    with torch.no_grad():
        y = adapter(x)

    max_error = (y - x).abs().max().item()

    print(f"max(|y-x|) = {max_error:.3e}")
    assert y.shape == x.shape
    assert max_error < 1e-7
    # gamma = 0
    assert adapter.gamma.item() == 0.0

    # y = x + 0 * delta_x
    torch.testing.assert_close(
        y,
        x,
        rtol=0.0,
        atol=0.0,
    )


def test_gamma_is_trainable():
    adapter = ResidualAdapter2d(
        in_channels=64,
        bottleneck_channels=16,
        gamma_init=0.0,
    )

    x = torch.randn(
        4, 64, 8, 8,
        requires_grad=True,
    )

    target = torch.randn_like(x)

    y = adapter(x)

    loss = torch.nn.functional.mse_loss(
        y,
        target,
    )

    loss.backward()

    assert adapter.gamma.grad is not None
    assert torch.isfinite(adapter.gamma.grad).all()


def test_nonzero_gamma_changes_features():
    adapter = ResidualAdapter2d(
        in_channels=64,
        bottleneck_channels=16,
        gamma_init=1.0,
    )

    x = torch.randn(2, 64, 8, 8)

    with torch.no_grad():
        y = adapter(x)

    assert not torch.equal(x, y)