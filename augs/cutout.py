import numpy as np
import torch


class cutout:
    def __init__(self, perc=0.1):
        self.perc = perc

    def __call__(self, x):
        # x: [B, T, D]
        T = x.size(1)
        new_x = x.clone()
        win_len = int(self.perc * T)
        if win_len <= 0 or win_len >= T:
            return new_x
        start = np.random.randint(0, T - win_len)
        new_x[:, start : start + win_len, :] = 0.0
        return new_x
