import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AdaptiveWavelet(nn.Module):
    """Learnable chirplet atoms: Gaussian envelope × frequency-modulated cosine."""

    def __init__(self, n_vars, n_atoms):
        super().__init__()
        self.freq_params = nn.Parameter(torch.randn(n_vars, n_atoms, 3))

    def forward(self, x):
        # x: [B, T, N]
        B, T, N = x.shape
        t = torch.linspace(0, 1, T, device=x.device)
        t2 = t ** 2

        alpha = self.freq_params[..., 0].unsqueeze(-1)  # [N, K, 1]
        beta  = self.freq_params[..., 1].unsqueeze(-1)
        gamma = self.freq_params[..., 2].unsqueeze(-1)

        atoms = torch.exp(-alpha * t2) * torch.cos(beta * t + gamma * t2)  # [N, K, T]
        coeffs = torch.einsum("btn,nkt->bktn", x, atoms)                   # [B, K, T, N]
        return coeffs, atoms


class CoherenceAttention(nn.Module):
    """Cross-temporal spectral coherence attention."""

    def __init__(self, d_model, n_atoms, hidden_dim):
        super().__init__()
        self.n_atoms = n_atoms
        self.hidden_dim = hidden_dim
        self.scale = n_atoms ** -0.5

        self.proj    = nn.Linear(d_model, hidden_dim * 3)
        self.var_attn = nn.Parameter(torch.eye(d_model))
        self.var_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Linear(hidden_dim, d_model)
        self.dropout  = nn.Dropout(0.1)

    def forward(self, coeffs):
        # coeffs: [B, K, T, D]
        B, K, T, D = coeffs.shape
        H = self.hidden_dim

        qkv = self.proj(coeffs).reshape(B, K, T, H, 3)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]  # each [B, K, T, H]

        # rearrange to [B, K, H, T] so rfft runs on H
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        Q_fft = torch.fft.rfft(q, dim=2)         # [B, K, H//2+1, T]
        K_fft = torch.fft.rfft(k, dim=2)

        P_xy = (Q_fft * K_fft.conj()).mean(dim=2) # [B, K, T]
        P_xx = (Q_fft * Q_fft.conj()).mean(dim=2)
        P_yy = (K_fft * K_fft.conj()).mean(dim=2)
        coherence = P_xy.abs().pow(2) / (P_xx.abs() * P_yy.abs()).clamp(min=1e-6)

        time_attn = self.dropout(F.softmax(coherence / self.scale, dim=-1))  # [B, K, T]
        out = time_attn.unsqueeze(2) * v          # [B, K, H, T]

        # flatten K and H, mix variables, restore
        out = rearrange(out, "b k h t -> b t (k h)")       # [B, T, K*H]
        var_w = F.softmax(self.var_attn, dim=-1)
        out = torch.einsum("bld,cc->bld", out, var_w)      # scalar-scale by trace
        out = rearrange(out, "b t (k h) -> b t k h", k=K)  # [B, T, K, H]
        out = out + self.var_mlp(out)
        out = self.out_proj(out)                            # [B, T, K, D]
        return out.permute(0, 2, 1, 3)                     # [B, K, T, D]


class KoopmanLayer(nn.Module):
    """Unitary Koopman operator via QR decomposition (Stiefel manifold)."""

    def __init__(self, n_atoms):
        super().__init__()
        self.U     = nn.Parameter(torch.randn(n_atoms, n_atoms, dtype=torch.complex64))
        self.theta = nn.Parameter(torch.rand(n_atoms))
        with torch.no_grad():
            self.U.data = self.U / self.U.norm(dim=0, keepdim=True)

    def forward(self, x):
        # x: [B, K, T, N] complex
        B, K, T, N = x.shape
        U, _ = torch.linalg.qr(self.U)
        K_mat = U @ torch.diag(torch.exp(1j * self.theta)) @ U.conj().T  # [K, K]

        x = x.permute(0, 3, 1, 2).reshape(B * N, K, T)
        x = torch.bmm(K_mat.unsqueeze(0).expand(B * N, -1, -1), x)
        return x.reshape(B, N, K, T).permute(0, 2, 3, 1)  # [B, K, T, N]


class SonnetBlock(nn.Module):
    def __init__(self, d_model, n_atoms, downsample_factor=1):
        super().__init__()
        self.downsample = downsample_factor > 1
        self.pool       = nn.AvgPool1d(downsample_factor) if self.downsample else nn.Identity()
        self.wavelet    = AdaptiveWavelet(d_model, n_atoms)
        self.attention  = CoherenceAttention(d_model, n_atoms, d_model)
        self.koopman    = KoopmanLayer(n_atoms)

    def forward(self, x):
        # x: [B, T, d_model]
        if self.downsample:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)  # [B, T/f, d_model]

        coeffs, atoms = self.wavelet(x)                    # [B, K, T', d_model], [d_model, K, T']
        z = self.attention(coeffs)                         # [B, K, T', d_model]
        z = self.koopman(z.to(torch.complex64))            # [B, K, T', d_model]
        recon = torch.einsum("bktn,nkt->btn", z.real, atoms)  # [B, T', d_model]
        return recon


class Sonnet(nn.Module):

    optional = {
        "sonnet_d_model": 64,
        "n_atoms": 8,
        "sonnet_downsample": "1",  # comma-separated ints, e.g. "1,2,4"
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--sonnet_d_model", type=int, default=None)
        parser.add_argument("--n_atoms",         type=int, default=None)
        parser.add_argument("--sonnet_downsample", type=str, default=None)

    def __init__(self, configs):
        super().__init__()
        in_ch    = configs.input_channels
        out_ch   = configs.output_channels
        seq_len  = configs.input_len
        pred_len = configs.output_len
        d_model  = configs.sonnet_d_model
        n_atoms  = configs.n_atoms

        factors = [int(f) for f in str(configs.sonnet_downsample).split(",")]

        self.embed = nn.Linear(in_ch, d_model)
        self.blocks = nn.ModuleList([
            SonnetBlock(d_model=d_model, n_atoms=n_atoms, downsample_factor=f)
            for f in factors
        ])
        self.decoder = nn.Sequential(
            nn.Conv1d(seq_len,      pred_len * 4, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(pred_len * 4, pred_len * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(pred_len * 2, pred_len,     kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(pred_len),
        )
        self.project = nn.Linear(pred_len, out_ch)

    def forward(self, x, **kwargs):
        # x: [B, input_len, in_ch]
        mean = x.mean(dim=1, keepdim=True).detach()
        std  = x.std( dim=1, keepdim=True, unbiased=False).detach().clamp(min=1e-5)
        x = (x - mean) / std

        x = self.embed(x)  # [B, T, d_model]

        outs = []
        for block in self.blocks:
            out = block(x)  # [B, T' or T, d_model]
            if out.shape[1] != x.shape[1]:
                out = F.interpolate(
                    out.transpose(1, 2), size=x.shape[1], mode="linear", align_corners=False
                ).transpose(1, 2)
            outs.append(out)

        fused = torch.stack(outs, dim=0).mean(dim=0)  # [B, T, d_model]
        out = self.decoder(fused)                      # [B, pred_len, pred_len]
        out = self.project(out)                        # [B, pred_len, out_ch]

        out = out * std + mean                         # denorm (broadcasts over pred_len)
        return out
