import torch
from torch import Tensor


def thermometer(v: Tensor, n_bins: int = 50, vmin: float = 0, vmax: float = 1) -> Tensor:
    """Thermometer encoding of a scalar quantity.

    Parameters
    ----------
    v: Tensor
        Value(s) to encode. Can be any shape
    n_bins: int
        The number of dimensions to encode the values into
    vmin: float
        The smallest value, below which the encoding is equal to torch.zeros(n_bins)
    vmax: float
        The largest value, beyond which the encoding is equal to torch.ones(n_bins)
    Returns
    -------
    encoding: Tensor
        The encoded values, shape: `v.shape + (n_bins,)`
    """
    bins = torch.linspace(vmin, vmax, n_bins)
    gap = bins[1] - bins[0]
    assert gap > 0, "vmin and vmax must be different"
    # v[..., None] -> vの次元を一つ増やす
    # v.ndim -> vの次元数
    # (1,) * v.ndim -> 1をvの次元数だけ繰り返したリスト
    # (1,) * v.ndim + (-1,) -> 1をvの次元数だけ繰り返したリストに-1を足したリスト
    # bins.reshape((1,) * v.ndim + (-1,)) -> binsの次元をvの次元と同じにする
    # (v[..., None] - bins.reshape((1,) * v.ndim + (-1,))).clamp(0, gap.item()) -> vとbinsの差を計算し、0とgapの間にクランプ
    # / gap -> 0から1の間に正規化
    return (v[..., None] - bins.reshape((1,) * v.ndim + (-1,))).clamp(0, gap.item()) / gap

if __name__ == "__main__":
    v = torch.tensor([64.0, 64.0, 64.0])
    print(thermometer(v, n_bins=32, vmin=0, vmax=1))
