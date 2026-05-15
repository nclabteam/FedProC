import torch
import torch.nn as nn

from layers.DWT_Decomposition import Decomposition


class _TokenMixer(nn.Module):
    def __init__(self, input_seq, pred_seq, dropout, factor):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_seq, pred_seq * factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_seq * factor, pred_seq),
        )

    def forward(self, x):
        return self.layers(x.transpose(1, 2)).transpose(1, 2)


class _Mixer(nn.Module):
    def __init__(self, input_seq, out_seq, channel, d_model, dropout, tfactor, dfactor):
        super().__init__()
        self.tMixer = _TokenMixer(input_seq, out_seq, dropout, tfactor)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm2d(channel)
        self.norm2 = nn.BatchNorm2d(channel)
        self.embeddingMixer = nn.Sequential(
            nn.Linear(d_model, d_model * dfactor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * dfactor, d_model),
        )

    def forward(self, x):
        # x: [B, channel, patch_num, d_model]
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)  # [B, d_model, channel, patch_num]
        x = self.dropout(self.tMixer(x))
        x = x.permute(0, 2, 3, 1)  # [B, channel, patch_num, d_model]
        x = self.norm2(x)
        x = x + self.dropout(self.embeddingMixer(x))
        return x


class _ResolutionBranch(nn.Module):
    def __init__(self, input_seq, pred_seq, channel, d_model, dropout, embedding_dropout, tfactor, dfactor, patch_len, patch_stride):
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_num = int((input_seq - patch_len) / patch_stride + 2)

        self.patch_norm = nn.BatchNorm2d(channel)
        self.patch_embedding_layer = nn.Linear(patch_len, d_model)
        self.mixer1 = _Mixer(self.patch_num, self.patch_num, channel, d_model, dropout, tfactor, dfactor)
        self.mixer2 = _Mixer(self.patch_num, self.patch_num, channel, d_model, dropout, tfactor, dfactor)
        self.norm = nn.BatchNorm2d(channel)
        self.emb_dropout = nn.Dropout(embedding_dropout)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * d_model, pred_seq),
        )

    def _do_patching(self, x):
        x_end = x[:, :, -1:]
        x = torch.cat((x, x_end.repeat(1, 1, self.patch_stride)), dim=-1)
        return x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)

    def forward(self, x):
        # x: [B, channel, input_seq]
        x_patch = self._do_patching(x)
        x_patch = self.patch_norm(x_patch)
        x_emb = self.emb_dropout(self.patch_embedding_layer(x_patch))
        out = self.mixer1(x_emb)
        out = self.norm(out + self.mixer2(out))
        return self.head(out)  # [B, channel, pred_seq]


class WPMixer(nn.Module):
    """WPMixer: Wavelet Packet Mixer for Time Series Forecasting. arXiv 2025."""

    optional = {
        "d_model": 32,
        "dropout": 0.1,
        "patch_len": 8,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--d_model", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)
        parser.add_argument("--patch_len", type=int, default=None)

    def __init__(self, configs, tfactor=5, dfactor=5, wavelet="db2", level=1, stride=8):
        super().__init__()
        self.pred_len = configs.output_len
        channel = configs.output_channels
        d_model = configs.d_model
        dropout = configs.dropout
        patch_len = configs.patch_len
        patch_stride = stride

        self.decomp = Decomposition(configs.input_len, configs.output_len, wavelet_name=wavelet, level=level)

        self.branches = nn.ModuleList(
            [
                _ResolutionBranch(
                    self.decomp.input_w_dim[i],
                    self.decomp.pred_w_dim[i],
                    channel,
                    d_model,
                    dropout,
                    dropout,
                    tfactor,
                    dfactor,
                    patch_len,
                    patch_stride,
                )
                for i in range(len(self.decomp.input_w_dim))
            ]
        )

    def forward(self, x, **kwargs):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        x_t = x.transpose(1, 2)  # [B, channel, input_len]
        yl, yh = self.decomp.transform(x_t)

        pred_yl = self.branches[0](yl)
        pred_yh = [self.branches[i + 1](yh[i]) for i in range(len(yh))]

        out = self.decomp.inv_transform(pred_yl, pred_yh)  # [B, channel, pred_len+]
        out = out.transpose(1, 2)[:, : self.pred_len, :]  # [B, pred_len, channel]

        out = out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        out = out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return out
