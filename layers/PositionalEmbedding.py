import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Source:
        https://github.com/huangst21/TimeKAN/blob/main/layers/Embed.py
        https://github.com/thuml/Large-Time-Series-Model/blob/main/layers/Embed.py
        https://github.com/XiangMa-Shaun/U-Mixer/blob/main/layers/Embed.py
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space.
        # pe: (max_len, d_model) -> expanded to (1, max_len, d_model) before register
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        # position: (max_len, 1) - each row is the position index
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # div_term: (d_model/2,) - frequencies for even indices
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        # pe[:, 0::2] selects even dimensions -> shape (max_len, d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] selects odd dimensions  -> shape (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)

        # add batch dim: pe -> (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: expected shape (batch, seq_len, ...)
        # returned tensor: pe[:, : x.size(1)] has shape (1, seq_len, d_model)
        # typically you will broadcast or add this to input embeddings of shape (batch, seq_len, d_model)
        return self.pe[:, : x.size(1)]
