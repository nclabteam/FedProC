import random

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


class _Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size]


class _CausalConvBlock(nn.Module):
    """Two causal dilated convolutions with residual connection."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation, final=False):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size, padding=pad, dilation=dilation
        )
        self.chomp1 = _Chomp1d(pad)
        self.drop1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size, padding=pad, dilation=dilation
        )
        self.chomp2 = _Chomp1d(pad)
        self.drop2 = nn.Dropout(0.1)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU() if final else None

    def forward(self, x):
        out = self.drop1(F.gelu(self.chomp1(self.conv1(x))))
        out = self.drop2(F.gelu(self.chomp2(self.conv2(out))))
        res = x if self.residual is None else self.residual(x)
        return self.relu(out + res) if self.relu is not None else out + res


class _CausalCNNEncoder(nn.Module):
    """
    Multi-scale causal CNN encoder.

    Takes [B, T, C] time series and produces:
      - trend:      [B, T, repr_dim]   temporal representations
      - trend_repr: [B, repr_dim]      last-step summary
    """

    def __init__(self, in_ch, repr_dim, kernel_list):
        super().__init__()
        self.kernel_list = kernel_list
        self.input_fc = _CausalConvBlock(in_ch, repr_dim, 1, 1)
        self.repr_dropout = nn.Dropout(0.1)
        self.multi_cnn = nn.ModuleList(
            [nn.Conv1d(repr_dim, repr_dim, k, padding=k - 1) for k in kernel_list]
        )

    def _encode(self, x):
        # x: [B, T, C]
        x = x.transpose(2, 1)  # [B, C, T]
        x = self.input_fc(x)  # [B, repr_dim, T]
        parts = []
        for idx, mod in enumerate(self.multi_cnn):
            out = mod(x)  # [B, repr_dim, T + pad]
            if self.kernel_list[idx] != 1:
                out = out[..., : -(self.kernel_list[idx] - 1)]
            parts.append(out.transpose(1, 2))  # [B, T, repr_dim]
        trend = reduce(
            rearrange(parts, "list b t d -> list b t d"),
            "list b t d -> b t d",
            "mean",
        )
        return self.repr_dropout(trend)  # [B, T, repr_dim]

    def forward(self, x_h, x_f=None, train=True):
        trend_h = self._encode(x_h)
        if not train:
            return trend_h, trend_h[:, -1, :], None
        # detach future so gradients only flow through the predictor
        trend_f = self._encode(x_f).detach()
        return trend_h, trend_h[:, -1, :], trend_f


class _Predictor(nn.Module):
    """Maps a history summary vector to a sequence of future representations."""

    def __init__(self, repr_dim, timestep):
        super().__init__()
        self.wl = nn.Linear(1, timestep)
        self.wl2 = nn.Linear(repr_dim, repr_dim)
        self.drop = nn.Dropout(0.25)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, repr_dim, 1]
        x = self.relu(self.wl(x)).transpose(1, 2)  # [B, timestep, repr_dim]
        return self.wl2(x)  # [B, timestep, repr_dim]


class SimTS(nn.Module):
    """
    SimTS: Similarity-based Representation Learning for Time Series.

    Architecture:
      - Encoder: multi-scale causal CNN that maps the input context window to
        temporal representations.
      - Predictor: a two-layer linear network that predicts future representations
        from the last history embedding (used only during self-supervised pretraining).
      - Head: a linear projection from the encoder summary to the forecast horizon.

    Training is two-phase (see SimTS strategy):
      Phase 1 – Self-supervised pretraining with a cosine similarity loss between
                predicted and encoded future representations.
      Phase 2 – Supervised fine-tuning of the full model on MSE forecasting loss.

    Reference: Zheng & Ma, "SimTS: Rethinking Contrastive Representation Learning
    for Time Series Forecasting", arXiv:2303.18205. Short version accepted at
    ICASSP 2024.
    """

    optional = {
        "simts_repr_dim": 64,
        "simts_K": 50,
        "simts_kernel_list": "1,2,4,8,16,32",
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--simts_repr_dim", type=int, default=None)
        parser.add_argument("--simts_K", type=int, default=None)
        parser.add_argument("--simts_kernel_list", type=str, default=None)

    def __init__(self, configs):
        super().__init__()
        in_ch = configs.input_channels
        out_ch = configs.output_channels
        seq_len = configs.input_len
        pred_len = configs.output_len

        repr_dim = configs.simts_repr_dim
        kernel_list = [int(k) for k in configs.simts_kernel_list.split(",")]

        # K must be strictly less than seq_len to leave room for the future segment
        self.K = min(configs.simts_K, seq_len - 1)
        self.timestep = seq_len - self.K
        self._pred_len = pred_len
        self._out_ch = out_ch

        self.encoder = _CausalCNNEncoder(in_ch, repr_dim, kernel_list)
        self.predictor = _Predictor(repr_dim, self.timestep)
        self.head = nn.Linear(repr_dim, pred_len * out_ch)

    # ------------------------------------------------------------------
    # Self-supervised pre-training objective
    # ------------------------------------------------------------------

    def pretrain_loss(self, x):
        """Cosine similarity loss between predicted and encoded future reps.

        Args:
            x: [B, seq_len, in_ch]
        Returns:
            scalar loss tensor
        """
        x1 = x[:, : self.K, :]  # history  [B, K, in_ch]
        x2 = x[:, self.K :, :]  # future   [B, timestep, in_ch]

        z1, _, z2 = self.encoder(x1, x2, train=True)
        # pick a random history timestep as the summary to predict from
        rand_idx = random.randint(0, z1.shape[1] - 1)
        summary = z1[:, rand_idx, :]  # [B, repr_dim]

        fcst = self.predictor(summary.unsqueeze(-1))  # [B, timestep, repr_dim]
        # negative cosine similarity averaged over timestep and batch
        return -F.cosine_similarity(z2, fcst, dim=2).mean()

    # ------------------------------------------------------------------
    # Supervised forward pass
    # ------------------------------------------------------------------

    def forward(self, x, **kwargs):
        """Encode full context → forecast.

        Args:
            x: [B, seq_len, in_ch]
        Returns:
            [B, pred_len, out_ch]
        """
        _, repr_, _ = self.encoder(x, train=False)  # [B, repr_dim]
        out = self.head(repr_)  # [B, pred_len * out_ch]
        return out.view(x.shape[0], self._pred_len, self._out_ch)
