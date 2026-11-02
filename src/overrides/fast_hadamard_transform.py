import torch

# The Hadamard transform is a multiply by a fixed n x n matrix of +-1, so it is
# done as one fp32 matmul with that matrix, cached per (n, device). This replaced
# a 7-round butterfly (n = 128) that ran each round as a torch.cat into a fresh
# fp32 tensor: same transform, fp32 terms added in a different order. One
# 2048-token chunk's indexer q: 113 ms -> 17 ms on 88 threads; the 4K-prompt
# run's 16 tokens are identical (PF4KH vs PF4KF).
_H = {}


def _hadamard_matrix(n: int, device) -> torch.Tensor:
    """Sylvester [[H, H], [H, -H]], unscaled +-1, cached per (n, device).
    Natural row order (the order the butterfly produced) and symmetric, so
    x @ H is the transform of each row of x."""
    key = (n, str(device))
    H = _H.get(key)
    if H is None:
        H = torch.ones(1, 1, dtype=torch.float32, device=device)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _H[key] = H = H.contiguous()
    return H


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    n = x.shape[-1]
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("Hadamard transform requires a power-of-two last dimension")

    y = x.float().reshape(-1, n) @ _hadamard_matrix(n, x.device)
    if scale != 1.0:
        y.mul_(scale)
    return y.reshape(x.shape).to(x.dtype)
