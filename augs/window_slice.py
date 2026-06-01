import numpy as np
import torch
import torch.nn.functional as F


class window_slice:
    def __init__(self, reduce_ratio=0.5):
        self.reduce_ratio = reduce_ratio

    def __call__(self, x):
        # x: [B, T, D]
        B, T, D = x.size()
        target_len = int(np.ceil(self.reduce_ratio * T))
        if target_len >= T:
            return x.clone()

        starts = np.random.randint(low=0, high=T - target_len, size=B)
        cropped = torch.stack(
            [x[i, starts[i] : starts[i] + target_len, :] for i in range(B)], dim=0
        )

        # Interpolate back to original size T
        # cropped is [B, target_len, D] -> interpolate expects [B, D, target_len]
        cropped_t = cropped.transpose(1, 2)
        interpolated = F.interpolate(
            cropped_t, size=T, mode="linear", align_corners=False
        )
        return interpolated.transpose(1, 2)
