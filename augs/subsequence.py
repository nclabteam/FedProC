import numpy as np
import torch


class subsequence:
    def __call__(self, x):
        T = x.size(1)
        if T <= 2:
            return x.clone()
        crop_l = np.random.randint(low=2, high=T + 1)
        new_x = x.clone()
        start = np.random.randint(T - crop_l + 1)
        end = start + crop_l

        new_x[:, :start, :] = 0.0
        new_x[:, end:, :] = 0.0
        return new_x
