import torch


class time_warp:
    def __init__(self, n_speed_change=4, max_speed_ratio=2.0):
        self.n_speed_change = n_speed_change
        self.max_speed_ratio = max_speed_ratio

    def __call__(self, x):
        B, T, D = x.size()
        if T <= 4:
            return x.clone()

        ratios = torch.empty(B, self.n_speed_change, device=x.device).uniform_(
            1.0 / self.max_speed_ratio, self.max_speed_ratio
        )
        anchors = torch.zeros(B, self.n_speed_change + 1, device=x.device)
        anchors[:, 1:] = torch.cumsum(ratios, dim=1)
        anchors = anchors / anchors[:, -1:].clone()

        grid = torch.linspace(0, 1, T, device=x.device).view(1, T).expand(B, T)
        warped_coords = torch.zeros_like(grid)
        segment_width = 1.0 / self.n_speed_change

        for i in range(self.n_speed_change):
            mask = (grid >= i * segment_width) & (grid <= (i + 1) * segment_width)
            if i == self.n_speed_change - 1:
                mask = grid >= i * segment_width
            seg_grid = (grid - i * segment_width) / segment_width
            seg_start = anchors[:, i : i + 1]
            seg_end = anchors[:, i + 1 : i + 2]
            warped_coords = torch.where(
                mask, seg_start + seg_grid * (seg_end - seg_start), warped_coords
            )

        warped_indices = warped_coords * (T - 1)
        idx_low = torch.floor(warped_indices).long().clamp(0, T - 2)
        idx_high = idx_low + 1
        weight_high = warped_indices - idx_low.float()
        weight_low = 1.0 - weight_high

        idx_low_expanded = idx_low.unsqueeze(-1).expand(-1, -1, D)
        idx_high_expanded = idx_high.unsqueeze(-1).expand(-1, -1, D)

        x_low = torch.gather(x, 1, idx_low_expanded)
        x_high = torch.gather(x, 1, idx_high_expanded)

        return weight_low.unsqueeze(-1) * x_low + weight_high.unsqueeze(-1) * x_high


class magnitude_warp(time_warp):
    def __call__(self, x):
        # Transpose to warp magnitudes along features
        x_t = x.transpose(1, 2)
        warped_t = super().__call__(x_t)
        return warped_t.transpose(1, 2)
