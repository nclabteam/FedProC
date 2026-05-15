import math

import torch
import torch.nn as nn


class _FourierFilter(nn.Module):
    """Split time series into time-variant and time-invariant components via FFT masking."""

    def __init__(self, mask_spectrum):
        super().__init__()
        self.mask_spectrum = mask_spectrum

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        mask = torch.ones_like(xf)
        if self.mask_spectrum is not None and len(self.mask_spectrum) > 0:
            mask[:, self.mask_spectrum, :] = 0
        x_var = torch.fft.irfft(xf * mask, dim=1)
        x_inv = x - x_var
        return x_var, x_inv


class _MLP(nn.Module):
    def __init__(
        self,
        f_in,
        f_out,
        hidden_dim=128,
        hidden_layers=2,
        dropout=0.05,
        activation="tanh",
    ):
        super().__init__()
        act = nn.Tanh() if activation == "tanh" else nn.ReLU()
        layers = [nn.Linear(f_in, hidden_dim), act, nn.Dropout(dropout)]
        for _ in range(hidden_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), act, nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _KPLayerApprox(nn.Module):
    """Koopman operator via DMD with multi-step approximation."""

    def __init__(self):
        super().__init__()
        self.K = None
        self.K_step = None

    def forward(self, z, pred_len=1):
        B, input_len, E = z.shape
        x, y = z[:, :-1], z[:, 1:]
        self.K = torch.linalg.lstsq(x, y).solution
        if torch.isnan(self.K).any():
            self.K = torch.eye(E, device=z.device).unsqueeze(0).expand(B, -1, -1)
        z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)
        if pred_len <= input_len:
            self.K_step = torch.linalg.matrix_power(self.K, pred_len)
            if torch.isnan(self.K_step).any():
                self.K_step = (
                    torch.eye(E, device=z.device).unsqueeze(0).expand(B, -1, -1)
                )
            z_pred = torch.bmm(z[:, -pred_len:, :], self.K_step)
        else:
            self.K_step = torch.linalg.matrix_power(self.K, input_len)
            if torch.isnan(self.K_step).any():
                self.K_step = (
                    torch.eye(E, device=z.device).unsqueeze(0).expand(B, -1, -1)
                )
            temp, all_pred = z, []
            for _ in range(math.ceil(pred_len / input_len)):
                temp = torch.bmm(temp, self.K_step)
                all_pred.append(temp)
            z_pred = torch.cat(all_pred, dim=1)[:, :pred_len, :]
        return z_rec, z_pred


class _TimeVarKP(nn.Module):
    """Koopman predictor for time-variant component using sliding window DMD."""

    def __init__(
        self, enc_in, input_len, pred_len, seg_len, dynamic_dim, encoder, decoder
    ):
        super().__init__()
        self.enc_in = enc_in
        self.input_len = input_len
        self.pred_len = pred_len
        self.seg_len = seg_len
        self.freq = math.ceil(input_len / seg_len)
        self.step = math.ceil(pred_len / seg_len)
        self.padding_len = seg_len * self.freq - input_len
        self.dynamics = _KPLayerApprox()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        B, L, C = x.shape
        res = torch.cat((x[:, L - self.padding_len :, :], x), dim=1)
        res = res.chunk(self.freq, dim=1)
        res = torch.stack(res, dim=1).reshape(B, self.freq, -1)
        res = self.encoder(res)
        x_rec, x_pred = self.dynamics(res, self.step)
        x_rec = self.decoder(x_rec).reshape(B, self.freq, self.seg_len, self.enc_in)
        x_rec = x_rec.reshape(B, -1, self.enc_in)[:, : self.input_len, :]
        x_pred = self.decoder(x_pred).reshape(B, self.step, self.seg_len, self.enc_in)
        x_pred = x_pred.reshape(B, -1, self.enc_in)[:, : self.pred_len, :]
        return x_rec, x_pred


class _TimeInvKP(nn.Module):
    """Koopman predictor for time-invariant component with learnable operator."""

    def __init__(self, input_len, pred_len, dynamic_dim, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        K_init = torch.randn(dynamic_dim, dynamic_dim)
        U, _, V = torch.svd(K_init)
        self.K = nn.Linear(dynamic_dim, dynamic_dim, bias=False)
        self.K.weight.data = torch.mm(U, V.t())

    def forward(self, x):
        res = x.transpose(1, 2)
        res = self.encoder(res)
        res = self.K(res)
        res = self.decoder(res)
        return res.transpose(1, 2)


class Koopa(nn.Module):
    """Koopa: Learning Non-stationary Time Series with Koopman Predictors. NeurIPS 2023."""

    optional = {
        "d_model": 128,
        "d_ff": 64,
        "e_layers": 3,
        "dropout": 0.05,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--d_model", type=int, default=None)
        parser.add_argument("--d_ff", type=int, default=None)
        parser.add_argument("--e_layers", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)

    def __init__(self, configs):
        super().__init__()
        self.input_len = configs.input_len
        self.pred_len = configs.output_len
        enc_in = configs.input_channels
        dynamic_dim = configs.d_model
        hidden_dim = configs.d_ff
        hidden_layers = 2
        num_blocks = configs.e_layers
        dropout = configs.dropout
        alpha = 0.2

        self.num_blocks = num_blocks
        self.alpha = alpha
        self._mask_initialized = False
        self.register_buffer("mask_spectrum", None)

        seg_len = self.pred_len
        self.disentanglement = _FourierFilter(None)

        self.time_inv_encoder = _MLP(
            self.input_len, dynamic_dim, hidden_dim, hidden_layers, dropout, "relu"
        )
        self.time_inv_decoder = _MLP(
            dynamic_dim, self.pred_len, hidden_dim, hidden_layers, dropout, "relu"
        )
        self.time_inv_kps = nn.ModuleList(
            [
                _TimeInvKP(
                    self.input_len,
                    self.pred_len,
                    dynamic_dim,
                    self.time_inv_encoder,
                    self.time_inv_decoder,
                )
                for _ in range(num_blocks)
            ]
        )

        self.time_var_encoder = _MLP(
            seg_len * enc_in, dynamic_dim, hidden_dim, hidden_layers, dropout, "tanh"
        )
        self.time_var_decoder = _MLP(
            dynamic_dim, seg_len * enc_in, hidden_dim, hidden_layers, dropout, "tanh"
        )
        self.time_var_kps = nn.ModuleList(
            [
                _TimeVarKP(
                    enc_in,
                    self.input_len,
                    self.pred_len,
                    seg_len,
                    dynamic_dim,
                    self.time_var_encoder,
                    self.time_var_decoder,
                )
                for _ in range(num_blocks)
            ]
        )

    def _init_mask(self, x):
        with torch.no_grad():
            xf = torch.fft.rfft(x, dim=1)
            amps = abs(xf).mean(dim=0).mean(dim=1)
            k = max(1, int(amps.shape[0] * self.alpha))
            self.mask_spectrum = amps.topk(k).indices
            self.disentanglement.mask_spectrum = self.mask_spectrum
        self._mask_initialized = True

    def forward(self, x, **kwargs):
        if not self._mask_initialized:
            self._init_mask(x)

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x = x / stdev

        residual, forecast = x, None
        for i in range(self.num_blocks):
            time_var_input, time_inv_input = self.disentanglement(residual)
            time_inv_output = self.time_inv_kps[i](time_inv_input)
            time_var_backcast, time_var_output = self.time_var_kps[i](time_var_input)
            residual = residual - time_var_backcast
            if forecast is None:
                forecast = time_inv_output + time_var_output
            else:
                forecast = forecast + time_inv_output + time_var_output

        forecast = forecast * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        forecast = forecast + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return forecast
