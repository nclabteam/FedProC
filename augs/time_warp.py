import torch


class time_warp:
    """Warp time through random monotonic speed segments."""

    def __init__(
        self,
        n_speed_change: int = 4,
        max_speed_ratio: float = 2.0,
    ) -> None:
        self.n_speed_change = n_speed_change
        self.max_speed_ratio = max_speed_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.size()
        if sequence_length <= 4:
            return x.clone()

        ratios = torch.empty(
            size=(batch_size, self.n_speed_change),
            device=x.device,
            dtype=x.dtype,
        ).uniform_(1.0 / self.max_speed_ratio, self.max_speed_ratio)
        anchors = torch.zeros(
            size=(batch_size, self.n_speed_change + 1),
            device=x.device,
            dtype=x.dtype,
        )
        anchors[:, 1:] = torch.cumsum(input=ratios, dim=1)
        anchors = anchors / anchors[:, -1:].clone()

        grid = (
            torch.linspace(
                start=0,
                end=1,
                steps=sequence_length,
                device=x.device,
                dtype=x.dtype,
            )
            .view(1, sequence_length)
            .expand(batch_size, sequence_length)
        )
        warped_coords = torch.zeros_like(grid)
        segment_width = 1.0 / self.n_speed_change

        for segment in range(self.n_speed_change):
            mask = (grid >= segment * segment_width) & (
                grid <= (segment + 1) * segment_width
            )
            if segment == self.n_speed_change - 1:
                mask = grid >= segment * segment_width
            segment_grid = (grid - segment * segment_width) / segment_width
            segment_start = anchors[:, segment : segment + 1]
            segment_end = anchors[:, segment + 1 : segment + 2]
            warped_coords = torch.where(
                condition=mask,
                input=segment_start + segment_grid * (segment_end - segment_start),
                other=warped_coords,
            )

        warped_indices = warped_coords * (sequence_length - 1)
        idx_low = torch.floor(warped_indices).long().clamp(0, sequence_length - 2)
        idx_high = idx_low + 1
        weight_high = warped_indices - idx_low.float()
        weight_low = 1.0 - weight_high

        idx_low_expanded = idx_low.unsqueeze(-1).expand(-1, -1, channels)
        idx_high_expanded = idx_high.unsqueeze(-1).expand(-1, -1, channels)

        x_low = torch.gather(input=x, dim=1, index=idx_low_expanded)
        x_high = torch.gather(input=x, dim=1, index=idx_high_expanded)

        return weight_low.unsqueeze(-1) * x_low + weight_high.unsqueeze(-1) * x_high


class magnitude_warp(time_warp):
    """Apply time warping across the feature dimension."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        transposed = x.transpose(1, 2)
        warped = super().__call__(x=transposed)
        return warped.transpose(1, 2)
