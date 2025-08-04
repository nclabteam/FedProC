import math
from typing import Callable, Tuple, Union

import numba
import numpy as np
import torch
import torch.nn.utils.rnn as rnn
from numba import cuda
from torch.autograd import Function

MAX_THREADS_PER_BLOCK = 1024

T = torch.Tensor
PS = rnn.PackedSequence


class SoftDTW(torch.nn.Module):
    """
    Paper: https://arxiv.org/abs/1703.01541
    Source: https://github.com/toinsson/pysdtw/blob/main/pysdtw/__init__.py
    """

    def __init__(
        self,
        gamma: float = 0.001,
        dist_func: Callable = None,
        use_cuda: bool = True,
        bandwidth: int = None,
        reduction="mean",
    ):
        """
        Args:
            gamma (float): Regularization parameter, lower is less smoothed (closer to true DTW).
            dist_func (func): Distance function used in pointwise computation, default to L2 squared.
            use_cuda (bool): Flag to use GPU, default to True.
            bandwidth (int): Sakoe-Chiba type bandwith parameter, default to 0.
        """
        super(SoftDTW, self).__init__()
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.dist_func = dist_func if dist_func is not None else pairwise_l2_squared
        self.use_cuda = use_cuda
        self.dtw_func = SoftDTWcuda.apply if use_cuda else SoftDTWcpu.apply
        self.reduction = reduction

    def forward(self, X: Union[T, PS], Y: Union[T, PS]):
        """Compute the soft-DTW value between X and Y.

        Args:
            X (tensor or PackedSequence): input of size batch_size x seq_len_x x dims
            Y (tensor or PackedSequence): input of size batch_size x seq_len_y x dims

        Returns:
            The soft-DTW distance between X and Y of size batch_size.
        """
        X, Y, XY_lengths = _prepare_input(X, Y)
        XY_D = self.dist_func(X, Y)
        dtw = self.dtw_func(XY_D, XY_lengths, self.gamma, self.bandwidth)
        if self.reduction == "mean":
            return dtw.mean()
        elif self.reduction == "sum":
            return dtw.sum()
        else:
            return dtw


def _prepare_input(x: Union[T, PS], y: Union[T, PS]) -> Tuple[T, T, T]:
    """Prepare the inputs. PackedSequences are unpacked. The lengths of
    individual sequences in x and y are returned as a staked array of shape
    (batchx2). Batch size and outer dimension of x and y must be the same.
    """
    x, x_len = _unpack_sequence(x)
    y, y_len = _unpack_sequence(y)
    xy_len = torch.stack([x_len, y_len]).T.to(x.device)

    bx, _, dx = x.shape
    by, _, dy = y.shape
    assert (bx == by) and (dx == dy)

    return x, y, xy_len


def _unpack_sequence(x: Union[T, PS]) -> Tuple[T, T]:
    """Return an unpacked sequence and lengths of subsequences."""
    if isinstance(x, rnn.PackedSequence):
        x, x_len = rnn.pad_packed_sequence(x, batch_first=True)
    else:
        u, v = x.shape[:2]
        x_len = torch.tensor([v]).expand(u)
    return x, x_len


def pairwise_l2_squared(x: T, y: T) -> T:
    """Computes the pairwise distance matrix between x and y using the
    quadratic expansion. This limits the memory cost to the detriment of compute
    accuracy.
    """
    x_norm = (x**2).sum(-1).unsqueeze(-1)
    y_norm = (y**2).sum(-1).unsqueeze(-2)
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y.mT)
    return torch.clamp(dist, 0.0, torch.inf)


def pairwise_l2_squared_exact(x: T, y: T) -> T:
    """Computes the pairwise distance matrix between x and y. This formula
    incurs a high memory cost.
    """
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    x = x.unsqueeze(2).expand(-1, n, m, d)
    y = y.unsqueeze(1).expand(-1, n, m, d)
    return torch.pow(x - y, 2).sum(3)


class SoftDTWcpu(Function):
    """CPU implementation of the Soft-DTW algorithm."""

    @staticmethod
    def forward(ctx, D, lengths, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.tensor(gamma, dtype=dtype, device=dev)
        bandwidth = torch.tensor(bandwidth, dtype=dtype, device=dev)

        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()

        R = (
            torch.Tensor(compute_softdtw(D_, lengths.numpy(), g_, b_))
            .to(dev)
            .type(dtype)
        )

        ctx.save_for_backward(D, R, lengths, gamma, bandwidth)

        Ms, Ns = lengths[:, 0], lengths[:, 1]
        res = R[:, Ms, Ns].diag()

        return res

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, lengths, gamma, bandwidth = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        E = (
            torch.Tensor(compute_softdtw_backward(D_, R_, lengths.numpy(), g_, b_))
            .to(dev)
            .type(dtype)
        )
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None, None


@numba.jit(nopython=True)
def sakoe_chiba_condition(i, j, M, N, bandwidth):
    """Approximate Sakoe-Chiba band for non-squared matrix."""
    i_sc, j_sc = i, j
    if N > M:
        i_sc = i * N / M
    if N < M:
        j_sc = j * M / N
    return abs(i_sc - j_sc) > bandwidth > 0


@numba.jit(nopython=True, parallel=True)
def compute_softdtw(D, lengths, gamma, bandwidth):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    for b in numba.prange(B):
        N, M = lengths[b]
        for j in range(1, M + 1):
            for i in range(1, N + 1):

                if sakoe_chiba_condition(i, j, N, M, bandwidth):
                    continue

                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = -gamma * (np.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    return R


@numba.jit(nopython=True, parallel=True)
def compute_softdtw_backward(D_, R, lengths, gamma, bandwidth):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1 : N + 1, 1 : M + 1] = D_

    for k in numba.prange(B):

        Ni, Mi = lengths[k]
        E[k, Ni + 1, Mi + 1] = 1
        R[k, :, Mi + 1] = -np.inf
        R[k, Ni + 1, :] = -np.inf
        R[k, Ni + 1, Mi + 1] = R[k, Ni, Mi]

        for j in range(Mi, 0, -1):
            for i in range(Ni, 0, -1):

                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf

                if sakoe_chiba_condition(i, j, N, M, bandwidth):
                    continue

                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = (
                    E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
                )
    return E[:, 1 : N + 1, 1 : M + 1]


class SoftDTWcuda(Function):
    """CUDA implementation of the Soft-DTW algorithm."""

    @staticmethod
    def forward(ctx, D, lengths, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.tensor(gamma, dtype=dtype, device=dev)
        bandwidth = torch.tensor(bandwidth, dtype=dtype, device=dev)

        B, M, N = D.shape
        T = min(max(M, N), MAX_THREADS_PER_BLOCK)
        n_passes = max(M, N) // MAX_THREADS_PER_BLOCK + 1
        n_antidiag = M + N - 1

        R = torch.ones((B, M + 2, N + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0

        compute_softdtw_cuda[B, T](
            cuda.as_cuda_array(D.detach()),
            gamma.item(),
            bandwidth.item(),
            cuda.as_cuda_array(lengths),
            n_passes,
            n_antidiag,
            cuda.as_cuda_array(R),
        )

        ctx.save_for_backward(D, R.clone(), lengths, gamma, bandwidth)

        Ms, Ns = lengths[:, 0], lengths[:, 1]
        res = R[:, Ms, Ns].diag()

        return res

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, lengths, gamma, bandwidth = ctx.saved_tensors

        B, M, N = D.shape
        T = min(max(M, N), MAX_THREADS_PER_BLOCK)
        n_passes = max(M, N) // MAX_THREADS_PER_BLOCK + 1
        n_antidiag = M + N - 1

        D_ = torch.zeros((B, M + 2, N + 2), dtype=dtype, device=dev)
        D_[:, 1 : M + 1, 1 : N + 1] = D
        E = torch.zeros((B, M + 2, N + 2), dtype=dtype, device=dev)

        for Bi, (Mi, Ni) in enumerate(lengths):
            R[Bi, :, Ni + 1] = -math.inf
            R[Bi, Mi + 1, :] = -math.inf
            R[Bi, Mi + 1, Ni + 1] = R[Bi, Mi, Ni]
            E[Bi, Mi + 1, Ni + 1] = 1

        compute_softdtw_backward_cuda[B, T](
            cuda.as_cuda_array(D_),
            cuda.as_cuda_array(R),
            1.0 / gamma.item(),
            bandwidth.item(),
            cuda.as_cuda_array(lengths),
            n_passes,
            n_antidiag,
            cuda.as_cuda_array(E),
        )

        E = E[:, 1 : M + 1, 1 : N + 1]
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None, None


@cuda.jit
def sakoe_chiba_condition(i, j, M, N, bandwidth):
    """Approximate Sakoe-Chiba band for non-squared matrix."""
    i_sc, j_sc = i, j
    if N > M:
        i_sc = i * N / M
    if N < M:
        j_sc = j * M / N
    return abs(i_sc - j_sc) > bandwidth > 0


@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, mn, n_passes, n_antidiag, R):
    inv_gamma = 1.0 / gamma

    Bi = cuda.blockIdx.x
    Mi, Ni = mn[Bi]
    thread_id = cuda.threadIdx.x

    for a in range(n_antidiag):
        for p in range(n_passes):

            I = a - thread_id - p * MAX_THREADS_PER_BLOCK
            J = thread_id + p * MAX_THREADS_PER_BLOCK

            if (I + J == a) and (I < Mi and J < Ni) and (I > -1):
                i, j = I + 1, J + 1

                if sakoe_chiba_condition(i, j, Mi, Ni, bandwidth):
                    continue

                r0 = -R[Bi, i - 1, j - 1] * inv_gamma
                r1 = -R[Bi, i - 1, j] * inv_gamma
                r2 = -R[Bi, i, j - 1] * inv_gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma * (math.log(rsum) + rmax)
                R[Bi, i, j] = D[Bi, i - 1, j - 1] + softmin

        cuda.syncthreads()


@cuda.jit
def compute_softdtw_backward_cuda(
    D, R, inv_gamma, bandwidth, mn, n_passes, n_antidiag, E
):
    Bi = cuda.blockIdx.x
    Mi, Ni = mn[Bi]
    thread_id = cuda.threadIdx.x

    for a in range(n_antidiag):
        rev_a = n_antidiag - a - 1
        for p in range(n_passes):
            I = rev_a - thread_id - p * MAX_THREADS_PER_BLOCK
            J = thread_id + p * MAX_THREADS_PER_BLOCK

            if (I + J == rev_a) and (I < Mi and J < Ni) and (I > -1):
                i, j = I + 1, J + 1

                if math.isinf(R[Bi, i, j]):
                    R[Bi, i, j] = -math.inf

                if sakoe_chiba_condition(i, j, Mi, Ni, bandwidth):
                    continue

                a = math.exp(
                    (R[Bi, i + 1, j] - R[Bi, i, j] - D[Bi, i + 1, j]) * inv_gamma
                )
                b = math.exp(
                    (R[Bi, i, j + 1] - R[Bi, i, j] - D[Bi, i, j + 1]) * inv_gamma
                )
                c = math.exp(
                    (R[Bi, i + 1, j + 1] - R[Bi, i, j] - D[Bi, i + 1, j + 1])
                    * inv_gamma
                )
                E[Bi, i, j] = (
                    E[Bi, i + 1, j] * a + E[Bi, i, j + 1] * b + E[Bi, i + 1, j + 1] * c
                )

        cuda.syncthreads()
