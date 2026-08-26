import torch


class subsequence:
    """Keep one random subsequence and mask its surroundings."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.size(1)
        if sequence_length <= 2:
            return x.clone()
        crop_length = torch.randint(
            low=2,
            high=sequence_length + 1,
            size=(),
            device=x.device,
        )
        start = torch.floor(
            torch.rand(size=(), device=x.device) * (sequence_length - crop_length + 1)
        ).long()
        positions = torch.arange(end=sequence_length, device=x.device)
        keep = (positions >= start) & (positions < start + crop_length)
        return x * keep.view(1, -1, 1)
