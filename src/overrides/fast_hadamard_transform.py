import torch


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    n = x.shape[-1]
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("Hadamard transform requires a power-of-two last dimension")

    orig_dtype = x.dtype
    leading_shape = x.shape[:-1]
    y = x.float().reshape(-1, n)

    h = 1
    while h < n:
        y = y.reshape(y.shape[0], -1, 2 * h)
        a = y[:, :, :h]
        b = y[:, :, h : 2 * h]
        y = torch.cat((a + b, a - b), dim=-1)
        y = y.reshape(-1, n)
        h *= 2

    y = y.reshape(*leading_shape, n)
    if scale != 1.0:
        y = y * scale
    return y.to(orig_dtype)
