import torch


class cutout:
    """Zero one random contiguous temporal window."""

    def __init__(self, perc: float = 0.1) -> None:
        self.perc = perc

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.size(1)
        window_length = int(self.perc * sequence_length)
        if window_length <= 0 or window_length >= sequence_length:
            return x.clone()
        start = torch.randint(
            low=0,
            high=sequence_length - window_length + 1,
            size=(),
            device=x.device,
        )
        positions = torch.arange(end=sequence_length, device=x.device)
        keep = (positions < start) | (positions >= start + window_length)
        return x * keep.view(1, -1, 1)
