import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


class window_warp:
    """Warp one random window and restore the original sequence length."""

    def __init__(
        self,
        window_ratio: float = 0.3,
        scales: Sequence[float] = (0.5, 2.0),
    ) -> None:
        self.window_ratio = window_ratio
        self.scales = tuple(scales)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.size()
        warp_size = math.ceil(self.window_ratio * sequence_length)
        if warp_size <= 0 or warp_size >= sequence_length - 2:
            return x.clone()

        scale_indices = torch.randint(
            low=0,
            high=len(self.scales),
            size=(batch_size,),
        ).tolist()
        warp_scales = [self.scales[index] for index in scale_indices]
        window_starts = torch.randint(
            low=1,
            high=sequence_length - warp_size - 1,
            size=(batch_size,),
        ).tolist()

        outputs = []
        for index, (window_start, warp_scale) in enumerate(
            zip(window_starts, warp_scales)
        ):
            window_end = window_start + warp_size
            window = x[index : index + 1, window_start:window_end, :].transpose(
                1,
                2,
            )
            warped_window = F.interpolate(
                input=window,
                size=max(1, int(warp_size * warp_scale)),
                mode="linear",
                align_corners=False,
            )

            prefix = x[index : index + 1, :window_start, :].transpose(1, 2)
            suffix = x[index : index + 1, window_end:, :].transpose(1, 2)
            merged = torch.cat(tensors=(prefix, warped_window, suffix), dim=-1)
            restored = F.interpolate(
                input=merged,
                size=sequence_length,
                mode="linear",
                align_corners=False,
            )
            outputs.append(restored.transpose(1, 2))

        return torch.cat(tensors=outputs, dim=0)
