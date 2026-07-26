import torch
import torch.nn as nn

from layers.MultiHeadLayerNorm import MultiHeadLayerNorm
from layers.RevIN import RevIN
from layers.sLSTM import sLSTMBlock


class xLSTMMixer(nn.Module):
    """xLSTM-Mixer (Kraus et al., 2025) for multivariate long-term forecasting.

    RevIN normalizes the input, a channel-independent NLinear forecast
    performs the initial time mixing, and an up-projection lifts it to the
    sLSTM embedding dimension. Learnable soft-prompt tokens are optionally
    prepended along the variate axis. A single sLSTM stack -- striding over
    VARIATES rather than time, as in the paper -- is then applied twice with
    shared weights: once to the up-projected embedding and once to a second
    view with the embedding's feature order reversed (multi-view mixing).
    The two views are concatenated and linearly reconciled into the
    forecast, then RevIN is inverted.

    Built on this repo's own sLSTM cell/block (``layers/sLSTM.py``), already
    verified against the official ``xlstm`` package to machine precision --
    the reference implementation at
    https://github.com/mauricekraus/xlstm-mixer builds its sLSTM blocks from
    that same package, so both paths are equivalent by construction.

    Fidelity note: multi-view mixing reverses the order of the embedding's
    *features* (``torch.flip(x, [-1])`` in the reference code), not the
    order of variates, even though Sec. 3.3's prose ("ensembling over
    variate orderings") can be read either way -- resolved in favor of the
    reference implementation, `xlstm_mixer/models/xlstm_mixer.py`.
    """

    optional = {
        "embedding_dim": 256,
        "num_heads": 8,
        "num_blocks": 1,
        "num_mem_tokens": 0,
        "conv_kernel_size": 0,
        "dropout": 0.1,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--embedding_dim", type=int, default=None)
        parser.add_argument("--num_heads", type=int, default=None)
        parser.add_argument("--num_blocks", type=int, default=None)
        parser.add_argument(
            "--num_mem_tokens",
            type=int,
            default=None,
            help="Learnable soft-prompt tokens prepended to the variate sequence",
        )
        parser.add_argument("--conv_kernel_size", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)

    def __init__(self, configs):
        super().__init__()
        self.enc_in = configs.input_channels
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.embedding_dim = configs.embedding_dim
        self.num_mem_tokens = configs.num_mem_tokens

        self.revin = RevIN(self.enc_in, affine=False, stdev_detach=False)
        self.linear = nn.Linear(self.seq_len, self.pred_len)
        self.pre_encoding = nn.Linear(self.pred_len, self.embedding_dim)

        if self.num_mem_tokens > 0:
            self.mem_tokens = nn.Parameter(
                torch.randn(self.num_mem_tokens, self.embedding_dim) * 0.01
            )

        self.blocks = nn.ModuleList(
            [
                sLSTMBlock(
                    embedding_dim=self.embedding_dim,
                    num_heads=configs.num_heads,
                    conv_kernel_size=configs.conv_kernel_size,
                    dropout=configs.dropout,
                    block_idx=idx,
                    num_blocks=configs.num_blocks,
                )
                for idx in range(configs.num_blocks)
            ]
        )
        self.post_norm = MultiHeadLayerNorm(self.embedding_dim, num_heads=1)
        self.fc_view = nn.Linear(2 * self.embedding_dim, self.pred_len)

    def _stack(self, h):
        for block in self.blocks:
            h = block(h)
        return self.post_norm(h)

    def forward(self, x, **kwargs):
        # x: (batch, seq_len, enc_in)
        x = self.revin(x, "norm")

        seq_last = x[:, -1:, :].detach()
        x0 = self.linear((x - seq_last).permute(0, 2, 1)).permute(0, 2, 1) + seq_last
        h = self.pre_encoding(x0.permute(0, 2, 1))  # (batch, enc_in, embedding_dim)

        if self.num_mem_tokens > 0:
            mem = self.mem_tokens.unsqueeze(0).expand(h.size(0), -1, -1)
            h = torch.cat([mem, h], dim=1)

        y_fwd = self._stack(h)
        y_bwd = self._stack(h.flip(-1))
        y = torch.cat([y_fwd, y_bwd], dim=-1)

        if self.num_mem_tokens > 0:
            y = y[:, self.num_mem_tokens :, :]

        out = self.fc_view(y).permute(0, 2, 1)  # (batch, pred_len, enc_in)
        return self.revin(out, "denorm")
