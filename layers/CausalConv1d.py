import torch.nn as nn


class CausalConv1d(nn.Module):
    """Dimension-wise (depthwise) causal convolution over the time axis.

    Used by both xLSTM blocks with window size 4 (Beck et al., 2024,
    "xLSTM: Extended Long Short-Term Memory", Fig. 9 and Fig. 10). A kernel
    size of 0 turns the layer into a no-op, as in the reference implementation.
    """

    def __init__(self, hidden_size, kernel_size=4):
        super().__init__()
        assert kernel_size >= 0, "kernel_size must be >= 0"
        self.kernel_size = kernel_size
        self.conv = (
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                groups=hidden_size,
                padding=kernel_size - 1,
            )
            if kernel_size > 0
            else None
        )

    def forward(self, x):
        if self.conv is None:
            return x
        # (batch, seq_len, channels) -> conv over time -> drop the right pad so
        # step t never sees t+1
        y = self.conv(x.transpose(1, 2))
        if self.kernel_size > 1:
            y = y[..., : -(self.kernel_size - 1)]
        return y.transpose(1, 2)
