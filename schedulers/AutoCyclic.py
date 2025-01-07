import math

import torch
import torch.nn as nn
from torch.fft import irfft, rfft
from torch.optim.lr_scheduler import _LRScheduler

optional = {
    "max_lr_multiplier": 7,
    "cycle_size": 25,
}


def args_update(parser):
    parser.add_argument("--max_lr_multiplier", type=float, default=None)
    parser.add_argument("--cycle_size", type=int, default=None)


_NEXT_FAST_LEN = {}


def next_fast_len(size):
    """
    Returns the next largest number ``n >= size`` whose prime factors are all
    2, 3, or 5. These sizes are efficient for fast fourier transforms.
    Equivalent to :func:`scipy.fftpack.next_fast_len`.

    Implementation from pyro

    :param int size: A positive number.
    :returns: A possibly larger number.
    :rtype int:
    """
    try:
        return _NEXT_FAST_LEN[size]
    except KeyError:
        pass

    assert isinstance(size, int) and size > 0
    next_size = size
    while True:
        remaining = next_size
        for n in (2, 3, 5):
            while remaining % n == 0:
                remaining //= n
        if remaining == 1:
            _NEXT_FAST_LEN[size] = next_size
            return next_size
        next_size += 1


def autocorrelation(input, dim=0):
    """
    Computes the autocorrelation of samples at dimension ``dim``.

    Reference: https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation

    Implementation copied form `pyro <https://github.com/pyro-ppl/pyro/blob/dev/pyro/ops/stats.py>`_.

    :param torch.Tensor input: the input tensor.
    :param int dim: the dimension to calculate autocorrelation.
    :returns torch.Tensor: autocorrelation of ``input``.
    """
    # Adapted from Stan implementation
    # https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/autocorrelation.hpp
    N = input.size(dim)
    M = next_fast_len(N)
    M2 = 2 * M

    # transpose dim with -1 for Fourier transform
    input = input.transpose(dim, -1)

    # centering and padding x
    centered_signal = input - input.mean(dim=-1, keepdim=True)

    # Fourier transform
    freqvec = torch.view_as_real(rfft(centered_signal, n=M2))
    # take square of magnitude of freqvec (or freqvec x freqvec*)
    freqvec_gram = freqvec.pow(2).sum(-1)
    # inverse Fourier transform
    autocorr = irfft(freqvec_gram, n=M2)

    # truncate and normalize the result, then transpose back to original shape
    autocorr = autocorr[..., :N]
    autocorr = autocorr / torch.tensor(
        range(N, 0, -1), dtype=input.dtype, device=input.device
    )
    autocorr = autocorr / autocorr[..., :1]
    return autocorr.transpose(dim, -1)


class AutoCyclic(_LRScheduler):
    """
    Source: https://github.com/wtfish/AutoCyclic/blob/main/autoCyclic.ipynb
    """

    def __init__(self, optimizer, configs, last_epoch=-1):
        self.base_lr = configs.learning_rate
        self.max_lr = configs.learning_rate * configs.max_lr_multiplier
        self.step_size = configs.cycle_size
        self.data = None
        super(AutoCyclic, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size))
        x = abs(self.last_epoch / self.step_size - 2 * cycle + 1)
        lr = [
            self.base_lr
            + (self.max_lr - self.base_lr)
            * (1 + math.cos(math.pi * x))
            / 2
            * (1 + self.get_batch_variance())
            for _ in self.base_lrs
        ]
        return lr

    def get_batch_variance(self):
        # AutoCorrelation+sigmoid
        if self.data is None:
            return 1
        step_var = []
        for items in self.data:
            SGLayer = nn.Sigmoid()
            output = autocorrelation(items)
            output = torch.nan_to_num(output, nan=0)
            output = SGLayer(output)
            step_var.append(torch.var(output))
        tensor_batch_step_var = torch.tensor(step_var)
        mean_step = torch.mean(tensor_batch_step_var)
        batch_variance = mean_step.numpy()
        return batch_variance

    def set_batch_data(self, data):
        self.data = data
