import math

import torch
import torch.nn.functional as F


class window_slice:
    """Crop a random window and interpolate it to the original length."""

    def __init__(self, reduce_ratio: float = 0.5) -> None:
        self.reduce_ratio = reduce_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.size()
        target_length = math.ceil(self.reduce_ratio * sequence_length)
        if target_length >= sequence_length:
            return x.clone()

        starts = torch.randint(
            low=0,
            high=sequence_length - target_length + 1,
            size=(batch_size, 1),
            device=x.device,
        )
        indices = starts + torch.arange(end=target_length, device=x.device)
        cropped = torch.gather(
            input=x,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, channels),
        )
        interpolated = F.interpolate(
            input=cropped.transpose(1, 2),
            size=sequence_length,
            mode="linear",
            align_corners=False,
        )
        return interpolated.transpose(1, 2)
