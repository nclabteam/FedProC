import numpy as np
import torch
import torch.nn.functional as F


class window_warp:
    def __init__(self, window_ratio=0.3, scales=[0.5, 2.0]):
        self.window_ratio = window_ratio
        self.scales = scales

    def __call__(self, x):
        B, T, D = x.size()
        warp_scales = np.random.choice(self.scales, B)
        warp_size = int(np.ceil(self.window_ratio * T))
        if warp_size <= 0 or warp_size >= T - 2:
            return x.clone()

        window_starts = np.random.randint(low=1, high=T - warp_size - 1, size=B)
        window_ends = window_starts + warp_size

        rets = []
        for i in range(B):
            window_seg = x[i : i + 1, window_starts[i] : window_ends[i], :].transpose(
                1, 2
            )
            target_size = int(warp_size * warp_scales[i])
            if target_size <= 0:
                target_size = 1
            window_seg_inter = F.interpolate(
                window_seg, size=target_size, mode="linear", align_corners=False
            )

            start_seg = x[i : i + 1, : window_starts[i], :].transpose(1, 2)
            end_seg = x[i : i + 1, window_ends[i] :, :].transpose(1, 2)

            merged = torch.cat([start_seg, window_seg_inter, end_seg], dim=-1)
            merged_inter = F.interpolate(
                merged, size=T, mode="linear", align_corners=False
            )
            rets.append(merged_inter.transpose(1, 2))

        return torch.cat(rets, dim=0)
