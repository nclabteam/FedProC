import torch.nn as nn

from layers.mLSTM import mLSTMBlock
from layers.sLSTM import sLSTMBlock
from utils.parsing import str2bool


class xLSTM(nn.Module):
    """xLSTM (Beck et al., 2024) adapted to multivariate forecasting.

    Residually stacks pre-LayerNorm xLSTM blocks: mLSTM blocks use the paper's
    pre up-projection structure (Fig. 10) and sLSTM blocks the post
    up-projection structure (Fig. 9). ``--stack_pattern`` -- or
    ``--num_blocks`` with ``--slstm_at`` -- fixes the xLSTM[a:b] block ratio.

    Reimplemented in-repo rather than depending on the official ``xlstm``
    package, whose sLSTM submodule probes a CUDA toolkit at import time
    regardless of the backend requested. Cells, gating, normalization and
    initialization follow that reference implementation.

    Forecasting adaptation: the paper's token embedding and language-model
    head are replaced by a linear input projection and a linear head mapping
    the final hidden state to the whole prediction window, as in this repo's
    other recurrent models. The mLSTM recurrence is the sequential form rather
    than the chunkwise-parallel one, which App. A.3 shows is equivalent.
    """

    optional = {
        "embedding_dim": 128,
        "num_heads": 4,
        "num_blocks": 6,
        "slstm_at": [1],
        "stack_pattern": None,
        "proj_factor": 2.0,
        "conv_kernel_size": 4,
        "qkv_proj_blocksize": 4,
        "ffn_proj_factor": 1.3,
        "ffn_act_fn": "gelu",
        "round_proj_to": 64,
        "bias": False,
        "dropout": 0.0,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--embedding_dim", type=int, default=None)
        parser.add_argument("--num_heads", type=int, default=None)
        parser.add_argument("--num_blocks", type=int, default=None)
        parser.add_argument("--slstm_at", type=int, nargs="+", default=None)
        parser.add_argument(
            "--stack_pattern",
            type=str,
            default=None,
            help=(
                "Per-block cell pattern, e.g. 'mmsmmm' ('m'=mLSTM, 's'=sLSTM). "
                "Overrides --num_blocks/--slstm_at when given."
            ),
        )
        parser.add_argument(
            "--proj_factor",
            type=float,
            default=None,
            help="mLSTM block up-projection factor (paper: 2)",
        )
        parser.add_argument(
            "--conv_kernel_size",
            type=int,
            default=None,
            help="Causal convolution window; 0 disables it",
        )
        parser.add_argument("--qkv_proj_blocksize", type=int, default=None)
        parser.add_argument(
            "--ffn_proj_factor",
            type=float,
            default=None,
            help="sLSTM block gated-MLP projection factor (paper: 4/3)",
        )
        parser.add_argument(
            "--ffn_act_fn",
            type=str,
            default=None,
            choices=["gelu", "relu", "relu^2", "sigmoid", "swish", "selu"],
        )
        parser.add_argument(
            "--round_proj_to",
            type=int,
            default=None,
            help="Round up-projection dims up to a multiple of this (1 disables)",
        )
        parser.add_argument("--bias", type=str2bool, default=None)
        parser.add_argument("--dropout", type=float, default=None)

    # None-valued args are stripped before reaching the model, so any optional
    # defaulting to None needs a class-level fallback
    stack_pattern = None

    def __init__(self, configs):
        super(xLSTM, self).__init__()
        self.enc_in = configs.input_channels
        self.pred_len = configs.output_len
        self.embedding_dim = configs.embedding_dim

        stack_pattern = getattr(configs, "stack_pattern", None)
        if stack_pattern:
            pattern = stack_pattern.lower()
            assert set(pattern) <= {
                "m",
                "s",
            }, f"stack_pattern must only contain 'm'/'s', got {stack_pattern!r}"
        else:
            slstm_at = set(configs.slstm_at)
            pattern = "".join(
                "s" if i in slstm_at else "m" for i in range(configs.num_blocks)
            )
        self.stack_pattern = pattern
        num_blocks = len(pattern)

        self.input_proj = nn.Linear(self.enc_in, self.embedding_dim)
        self.blocks = nn.ModuleList(
            [
                (
                    sLSTMBlock(
                        embedding_dim=self.embedding_dim,
                        num_heads=configs.num_heads,
                        conv_kernel_size=configs.conv_kernel_size,
                        ffn_proj_factor=configs.ffn_proj_factor,
                        ffn_act_fn=configs.ffn_act_fn,
                        dropout=configs.dropout,
                        bias=configs.bias,
                        block_idx=idx,
                        num_blocks=num_blocks,
                        round_proj_to=configs.round_proj_to,
                    )
                    if cell == "s"
                    else mLSTMBlock(
                        embedding_dim=self.embedding_dim,
                        num_heads=configs.num_heads,
                        proj_factor=configs.proj_factor,
                        conv_kernel_size=configs.conv_kernel_size,
                        qkv_proj_blocksize=configs.qkv_proj_blocksize,
                        dropout=configs.dropout,
                        bias=configs.bias,
                        num_blocks=num_blocks,
                        round_proj_to=configs.round_proj_to,
                    )
                )
                for idx, cell in enumerate(pattern)
            ]
        )
        self.norm_out = nn.LayerNorm(self.embedding_dim, bias=False)
        self.fc_pred = nn.Linear(self.embedding_dim, self.pred_len * self.enc_in)

    def forward(self, x, **kwargs):
        batch_size = x.size(0)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        pred = self.fc_pred(self.norm_out(h[:, -1, :]))
        return pred.view(batch_size, self.pred_len, self.enc_in)
