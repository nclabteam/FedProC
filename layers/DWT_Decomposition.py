import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class Decomposition(nn.Module):
    """Wavelet packet decomposition/reconstruction for 1D time series.

    Original: https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer
    """

    def __init__(self, input_length, pred_length, wavelet_name="db2", level=1, **kwargs):
        super().__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level

        self.dwt = DWT1DForward(wave=wavelet_name, J=level)
        self.idwt = DWT1DInverse(wave=wavelet_name)

        self.input_w_dim = self._coeff_lengths(input_length)
        self.pred_w_dim = self._coeff_lengths(pred_length)

    def _coeff_lengths(self, length):
        wav = pywt.Wavelet(self.wavelet_name)
        filter_len = wav.dec_len
        lengths = []
        cur = length
        for _ in range(self.level):
            cur = pywt.dwt_coeff_len(cur, filter_len, mode="zero")
            lengths.append(cur)
        # returns [yl_len, yh[0]_len, yh[1]_len, ...]
        return [lengths[-1]] + lengths

    def transform(self, x):
        yl, yh = self.dwt(x)
        return yl, yh

    def inv_transform(self, yl, yh):
        return self.idwt((yl, yh))


# ---------------------------------------------------------------------------
# DWT 1D Forward / Inverse
# ---------------------------------------------------------------------------


class DWT1DForward(nn.Module):
    def __init__(self, J=1, wave="db1", mode="zero", use_amp=False):
        super().__init__()
        self.use_amp = use_amp
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0, h1 = wave.dec_lo, wave.dec_hi
        else:
            h0, h1 = wave[0], wave[1]
        filts = _prep_filt_afb1d(h0, h1)
        self.register_buffer("h0", filts[0])
        self.register_buffer("h1", filts[1])
        self.J = J
        self.mode = mode

    def forward(self, x):
        assert x.ndim == 3, "Expected (N, C, L)"
        highs = []
        x0 = x
        mode = _mode_to_int(self.mode)
        for j in range(self.J):
            x0, x1 = AFB1D.apply(x0, self.h0, self.h1, mode, self.use_amp)
            highs.append(x1)
        return x0, highs


class DWT1DInverse(nn.Module):
    def __init__(self, wave="db1", mode="zero", use_amp=False):
        super().__init__()
        self.use_amp = use_amp
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0, g1 = wave.rec_lo, wave.rec_hi
        else:
            g0, g1 = wave[0], wave[1]
        filts = _prep_filt_sfb1d(g0, g1)
        self.register_buffer("g0", filts[0])
        self.register_buffer("g1", filts[1])
        self.mode = mode

    def forward(self, coeffs):
        x0, highs = coeffs
        assert x0.ndim == 3, "Expected (N, C, L)"
        mode = _mode_to_int(self.mode)
        for x1 in highs[::-1]:
            if x1 is None:
                x1 = torch.zeros_like(x0)
            if x0.shape[-1] > x1.shape[-1]:
                x0 = x0[..., :-1]
            x0 = SFB1D.apply(x0, x1, self.g0, self.g1, mode, self.use_amp)
        return x0


# ---------------------------------------------------------------------------
# Custom autograd functions
# ---------------------------------------------------------------------------


class AFB1D(Function):
    @staticmethod
    def forward(ctx, x, h0, h1, mode, use_amp):
        mode_str = _int_to_mode(mode)
        x = x[:, :, None, :]
        h0 = h0[:, :, None, :]
        h1 = h1[:, :, None, :]
        ctx.save_for_backward(h0, h1)
        ctx.shape = x.shape[3]
        ctx.mode = mode_str
        ctx.use_amp = use_amp
        lohi = _afb1d(x, h0, h1, use_amp, mode=mode_str, dim=3)
        x0 = lohi[:, ::2, 0].contiguous()
        x1 = lohi[:, 1::2, 0].contiguous()
        return x0, x1

    @staticmethod
    def backward(ctx, dx0, dx1):
        dx = None
        if ctx.needs_input_grad[0]:
            h0, h1 = ctx.saved_tensors
            dx0 = dx0[:, :, None, :]
            dx1 = dx1[:, :, None, :]
            dx = _sfb1d(dx0, dx1, h0, h1, ctx.use_amp, mode=ctx.mode, dim=3)[:, :, 0]
            if dx.shape[2] > ctx.shape:
                dx = dx[:, :, : ctx.shape]
        return dx, None, None, None, None, None


class SFB1D(Function):
    @staticmethod
    def forward(ctx, low, high, g0, g1, mode, use_amp):
        mode_str = _int_to_mode(mode)
        low = low[:, :, None, :]
        high = high[:, :, None, :]
        g0 = g0[:, :, None, :]
        g1 = g1[:, :, None, :]
        ctx.mode = mode_str
        ctx.save_for_backward(g0, g1)
        ctx.use_amp = use_amp
        return _sfb1d(low, high, g0, g1, use_amp, mode=mode_str, dim=3)[:, :, 0]

    @staticmethod
    def backward(ctx, dy):
        dlow, dhigh = None, None
        if ctx.needs_input_grad[0]:
            g0, g1 = ctx.saved_tensors
            dy = dy[:, :, None, :]
            dx = _afb1d(dy, g0, g1, ctx.use_amp, mode=ctx.mode, dim=3)
            dlow = dx[:, ::2, 0].contiguous()
            dhigh = dx[:, 1::2, 0].contiguous()
        return dlow, dhigh, None, None, None, None, None


# ---------------------------------------------------------------------------
# Low-level filter-bank operations
# ---------------------------------------------------------------------------


def _mode_to_int(mode):
    return {"zero": 0, "symmetric": 1, "per": 2, "periodization": 2, "constant": 3, "reflect": 4, "replicate": 5, "periodic": 6}[mode]


def _int_to_mode(mode):
    return {0: "zero", 1: "symmetric", 2: "periodization", 3: "constant", 4: "reflect", 5: "replicate", 6: "periodic"}[mode]


def _roll(x, n, dim):
    if n < 0:
        n = x.shape[dim] + n
    if dim == 2 or dim == -2:
        return torch.cat((x[:, :, -n:], x[:, :, :-n]), dim=2)
    elif dim == 3 or dim == -1:
        return torch.cat((x[:, :, :, -n:], x[:, :, :, :-n]), dim=3)
    raise ValueError(f"Unsupported dim: {dim}")


def _mypad(x, pad, mode="constant", value=0):
    if mode == "symmetric":
        if pad[2] == 0 and pad[3] == 0:
            m1, m2 = pad[0], pad[1]
            l = x.shape[-1]
            xe = _reflect(np.arange(-m1, l + m2, dtype="int32"), -0.5, l - 0.5)
            return x[:, :, :, xe]
        elif pad[0] == 0 and pad[1] == 0:
            m1, m2 = pad[2], pad[3]
            l = x.shape[-2]
            xe = _reflect(np.arange(-m1, l + m2, dtype="int32"), -0.5, l - 0.5)
            return x[:, :, xe]
    elif mode in ("constant", "reflect", "replicate"):
        return F.pad(x, pad, mode, value)
    elif mode == "zero":
        return F.pad(x, pad)
    raise ValueError(f"Unknown pad type: {mode}")


def _afb1d(x, h0, h1, use_amp, mode="zero", dim=-1):
    C = x.shape[1]
    d = dim % 4
    s = (2, 1) if d == 2 else (1, 2)
    N = x.shape[d]
    L = h0.numel()
    L2 = L // 2
    shape = [1, 1, 1, 1]
    shape[d] = L
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)

    if mode in ("per", "periodization"):
        if x.shape[d] % 2 == 1:
            x = torch.cat((x, x[:, :, :, -1:] if d == 3 else x[:, :, -1:, :]), dim=d)
            N += 1
        x = _roll(x, -L2, dim=d)
        pad = (L - 1, 0) if d == 2 else (0, L - 1)
        lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        N2 = N // 2
        if d == 2:
            lohi[:, :, :L2] += lohi[:, :, N2 : N2 + L2]
            lohi = lohi[:, :, :N2]
        else:
            lohi[:, :, :, :L2] += lohi[:, :, :, N2 : N2 + L2]
            lohi = lohi[:, :, :, :N2]
    else:
        outsize = pywt.dwt_coeff_len(N, L, mode=mode if mode != "zero" else "zero")
        p = 2 * (outsize - 1) - N + L
        if mode == "zero":
            if p % 2 == 1:
                pad = (0, 0, 0, 1) if d == 2 else (0, 1, 0, 0)
                x = F.pad(x, pad)
            pad = (p // 2, 0) if d == 2 else (0, p // 2)
            lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        elif mode in ("symmetric", "reflect", "periodic"):
            pad = (0, 0, p // 2, (p + 1) // 2) if d == 2 else (p // 2, (p + 1) // 2, 0, 0)
            x = _mypad(x, pad=pad, mode=mode)
            lohi = F.conv2d(x, h, stride=s, groups=C)
        else:
            raise ValueError(f"Unknown pad type: {mode}")
    return lohi


def _sfb1d(lo, hi, g0, g1, use_amp, mode="zero", dim=-1):
    C = lo.shape[1]
    d = dim % 4
    s = (2, 1) if d == 2 else (1, 2)
    L = g0.numel()
    shape = [1, 1, 1, 1]
    shape[d] = L
    N = 2 * lo.shape[d]
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)
    g0 = torch.cat([g0] * C, dim=0)
    g1 = torch.cat([g1] * C, dim=0)

    if mode in ("per", "periodization"):
        y = F.conv_transpose2d(lo, g0, stride=s, groups=C) + F.conv_transpose2d(hi, g1, stride=s, groups=C)
        if d == 2:
            y[:, :, : L - 2] += y[:, :, N : N + L - 2]
            y = y[:, :, :N]
        else:
            y[:, :, :, : L - 2] += y[:, :, :, N : N + L - 2]
            y = y[:, :, :, :N]
        y = _roll(y, 1 - L // 2, dim=dim)
    else:
        pad = (L - 2, 0) if d == 2 else (0, L - 2)
        y = F.conv_transpose2d(lo, g0, stride=s, padding=pad, groups=C) + F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=C)
    return y


def _prep_filt_afb1d(h0, h1, device=None):
    h0 = np.array(h0[::-1]).ravel()
    h1 = np.array(h1[::-1]).ravel()
    t = torch.get_default_dtype()
    h0 = torch.tensor(h0, device=device, dtype=t).reshape((1, 1, -1))
    h1 = torch.tensor(h1, device=device, dtype=t).reshape((1, 1, -1))
    return h0, h1


def _prep_filt_sfb1d(g0, g1, device=None):
    g0 = np.array(g0).ravel()
    g1 = np.array(g1).ravel()
    t = torch.get_default_dtype()
    g0 = torch.tensor(g0, device=device, dtype=t).reshape((1, 1, -1))
    g1 = torch.tensor(g1, device=device, dtype=t).reshape((1, 1, -1))
    return g0, g1


def _reflect(x, minx, maxx):
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    return np.array(np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx, dtype=x.dtype)
