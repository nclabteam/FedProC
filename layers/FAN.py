import torch
import torch.nn as nn


class FAN(nn.Module):
    """Separate dominant frequencies from residual forecasting signals."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        freq_topk: int = 20,
        rfft: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.epsilon = 1e-8
        self.freq_topk = freq_topk
        # print("freq_topk : ", self.freq_topk)
        self.rfft = rfft

        self.model_freq = MLPfreq(
            seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in
        )
        self.weight = nn.Parameter(torch.ones(2, self.enc_in))

    def loss(self, true: torch.Tensor) -> torch.Tensor:
        # freq normalization
        B, O, N = true.shape
        residual, pred_main = main_freq_part(
            x=true,
            k=self.freq_topk,
            rfft=self.rfft,
        )

        lf = nn.functional.mse_loss
        return lf(self.pred_main_freq_signal, pred_main) + lf(
            residual, self.pred_residual
        )

    def normalize(self, input: torch.Tensor) -> torch.Tensor:
        # (B, T, N)
        bs, len, dim = input.shape
        norm_input, x_filtered = main_freq_part(
            x=input,
            k=self.freq_topk,
            rfft=self.rfft,
        )
        self.pred_main_freq_signal = self.model_freq(
            main_freq=x_filtered.transpose(1, 2),
            x=input.transpose(1, 2),
        ).transpose(1, 2)

        return norm_input.reshape(bs, len, dim)

    def denormalize(self, input_norm: torch.Tensor) -> torch.Tensor:
        # input:  (B, O, N)
        # station_pred: outputs of normalize
        bs, len, dim = input_norm.shape
        # freq denormalize
        self.pred_residual = input_norm
        output = self.pred_residual + self.pred_main_freq_signal

        return output.reshape(bs, len, dim)

    def forward(
        self,
        batch_x: torch.Tensor,
        mode: str = "norm",
    ) -> torch.Tensor | None:
        if mode == "norm":
            return self.normalize(batch_x)
        elif mode == "denorm":
            return self.denormalize(batch_x)


def main_freq_part(
    x: torch.Tensor,
    k: int,
    rfft: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a signal into residual and top-frequency components."""

    # freq normalization
    if rfft:
        xf = torch.fft.rfft(x, dim=1)
    else:
        xf = torch.fft.fft(x, dim=1)

    k = min(k, xf.shape[1])
    k_values = torch.topk(xf.abs(), k, dim=1)
    indices = k_values.indices

    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    xf_filtered = xf * mask

    if rfft:
        x_filtered = torch.fft.irfft(xf_filtered, dim=1).real.float()
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()

    norm_input = x - x_filtered
    return norm_input, x_filtered


class MLPfreq(nn.Module):
    """Forecast dominant frequencies with a compact MLP."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in

        self.model_freq = nn.Sequential(
            nn.Linear(self.seq_len, 64),
            nn.ReLU(),
        )

        self.model_all = nn.Sequential(
            nn.Linear(64 + seq_len, 128), nn.ReLU(), nn.Linear(128, pred_len)
        )

    def forward(
        self,
        main_freq: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        inp = torch.concat([self.model_freq(main_freq), x], dim=-1)
        return self.model_all(inp)
