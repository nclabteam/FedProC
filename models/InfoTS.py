import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from augs import (
    cutout,
    jitter,
    scaling,
    subsequence,
    time_warp,
    window_slice,
    window_warp,
)

# ------------------------------------------------------------------
# Backbone Components
# ------------------------------------------------------------------


class SamePadConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1
    ):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride,
            groups=groups,
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation, stride=1, final=False
    ):
        super().__init__()
        self.conv1 = SamePadConv(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )
        self.conv2 = SamePadConv(
            out_channels, out_channels, kernel_size, stride=1, dilation=dilation
        )
        if stride == 1:
            self.projector = (
                nn.Conv1d(in_channels, out_channels, 1)
                if in_channels != out_channels or final
                else None
            )
        else:
            self.projector = nn.Conv1d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            *[
                ConvBlock(
                    channels[i - 1] if i > 0 else in_channels,
                    channels[i],
                    kernel_size=kernel_size,
                    dilation=2**i,
                    stride=stride,
                    final=(i == len(channels) - 1),
                )
                for i in range(len(channels))
            ]
        )

    def forward(self, x):
        return self.net(x)


class InfoTSShared:
    """Static paper formulations shared by the InfoTS components."""

    @staticmethod
    def continuous_mask(batch_size, length, count=5, span=0.1):
        mask = torch.ones(batch_size, length, dtype=torch.bool)
        count = int(count * length) if isinstance(count, float) else count
        count = max(min(count, length // 2), 1)
        span = int(span * length) if isinstance(span, float) else span
        span = max(span, 1)
        for row in range(batch_size):
            for _ in range(count):
                start = np.random.randint(length - span + 1)
                mask[row, start : start + span] = False
        return mask

    @staticmethod
    def binomial_mask(batch_size, length, probability=0.5):
        return torch.from_numpy(
            np.random.binomial(1, probability, size=(batch_size, length))
        ).to(torch.bool)

    @staticmethod
    def global_info_nce(raw, augmented, temperature=1.0):
        raw = raw.amax(dim=1)
        augmented = augmented.amax(dim=1)
        logits = raw @ augmented.T / temperature
        return F.cross_entropy(logits, torch.arange(raw.shape[0], device=raw.device))

    @staticmethod
    def local_info_nce(augmented, temperature=1.0, segments=8):
        batch_size, length, feature_dim = augmented.shape
        segments = min(segments, length)
        if segments < 2:
            return augmented.new_zeros(())
        segment_length = length // segments
        cropped = augmented[:, : segment_length * segments].reshape(
            batch_size, segments, segment_length, feature_dim
        )
        pooled = cropped.amax(dim=2)
        similarity = pooled @ pooled.transpose(1, 2) / temperature
        losses = []
        for index in range(segments):
            positive = index + 1 if index + 1 < segments else index - 1
            candidates = [positive] + [
                other for other in range(segments) if abs(other - index) > 1
            ]
            logits = similarity[:, index, candidates]
            losses.append(
                F.cross_entropy(
                    logits,
                    torch.zeros(batch_size, dtype=torch.long, device=logits.device),
                )
            )
        return torch.stack(losses).mean()

    @staticmethod
    def l1out(raw, augmented):
        if raw.shape[0] < 2:
            return raw.new_zeros(())
        raw = raw.amax(dim=1)
        augmented = augmented.amax(dim=1)
        similarity = raw @ augmented.T
        positive = similarity.diagonal()
        negatives = similarity.masked_fill(
            torch.eye(raw.shape[0], dtype=torch.bool, device=raw.device),
            float("-inf"),
        )
        return (positive - torch.logsumexp(negatives, dim=1)).mean()


# ------------------------------------------------------------------
# TS Encoder Wrapper
# ------------------------------------------------------------------


class TSEncoder(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_dims=64,
        depth=10,
        mask_mode="binomial",
        dropout=0.1,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims, [hidden_dims] * depth + [output_dims], kernel_size=3
        )
        self.repr_dropout = None if dropout == 0.0 else nn.Dropout(p=dropout)

    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(dim=-1)
        x = x.clone()
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = "all_true"

        if mask == "binomial":
            mask = InfoTSShared.binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "continuous":
            mask = InfoTSShared.continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "all_true":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == "all_false":
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == "mask_last":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # Conv encoder expects B x Ch x T
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        if self.repr_dropout is not None:
            x = self.repr_dropout(x)  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        return x


# ------------------------------------------------------------------
# Auto-Augmentation Module
# ------------------------------------------------------------------


class AutoAUG(nn.Module):
    """Paper Eq. 9-10 differentiable candidate-augmentation selector."""

    def __init__(self):
        super().__init__()
        self.augs = [
            subsequence(),
            cutout(),
            jitter(),
            scaling(),
            time_warp(),
            window_slice(),
            window_warp(),
        ]
        self.weight = nn.Parameter(torch.empty(len(self.augs)))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def get_sampling(self, temperature=1.0):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.training:
            epsilon = torch.rand_like(self.weight).clamp_(1e-4, 1 - 1e-4)
            return torch.sigmoid((torch.logit(epsilon) + self.weight) / temperature)
        return torch.sigmoid(self.weight)

    def forward(self, x, temperature=1.0):
        probabilities = self.get_sampling(temperature)
        candidates = [
            x + probability * (augmentation(x) - x)
            for probability, augmentation in zip(probabilities, self.augs)
        ]
        return torch.stack(candidates).mean(dim=0), x.clone()


# ------------------------------------------------------------------
# Main InfoTS Module
# ------------------------------------------------------------------


class InfoTS(InfoTSShared, nn.Module):
    optional = {
        "infots_repr_dim": 320,
        "infots_hidden_dim": 64,
        "infots_depth": 10,
        "infots_beta": 1.0,
        "infots_meta_beta": 1.0,
        "infots_k": 8,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--infots_repr_dim", type=int, default=None)
        parser.add_argument("--infots_hidden_dim", type=int, default=None)
        parser.add_argument("--infots_depth", type=int, default=None)
        parser.add_argument("--infots_beta", type=float, default=None)
        parser.add_argument("--infots_meta_beta", type=float, default=None)
        parser.add_argument("--infots_k", type=int, default=None)

    def __init__(self, configs):
        super().__init__()
        in_ch = configs.input_channels
        out_ch = configs.output_channels
        pred_len = configs.output_len

        repr_dim = configs.infots_repr_dim
        hidden_dim = configs.infots_hidden_dim
        depth = configs.infots_depth

        self.beta = configs.infots_beta
        self.meta_beta = configs.infots_meta_beta
        self.k = configs.infots_k

        # Dilated Conv backbone encoder
        self.encoder = TSEncoder(
            input_dims=in_ch,
            output_dims=repr_dim,
            hidden_dims=hidden_dim,
            depth=depth,
        )

        self.aug = AutoAUG()

        # Supervised mapping head for forecasting
        self.head = nn.Linear(repr_dim, pred_len * out_ch)
        self._pred_len = pred_len
        self._out_ch = out_ch

        # Unsupervised/Supervised classifier head used for the Meta-Update
        self.meta_unsup_head = nn.Linear(repr_dim, configs.batch_size)

    def get_features(self, x, temperature=1.0):
        a1, a2 = self.aug(x, temperature=temperature)
        out1 = self.encoder(a1)
        out2 = self.encoder(a2)
        return out1, out2

    # ------------------------------------------------------------------
    # Self-supervised pre-training contrastive objective
    # ------------------------------------------------------------------

    def pretrain_loss(self, x, temperature=1.0):
        """Computes InfoTS contrastive pretraining loss on the device.

        Args:
            x: [B, T, D] tensor
            temperature: Gumbel-Softmax temperature for AutoAUG sampling
        Returns:
            scalar loss tensor
        """
        augmented, raw = self.get_features(x, temperature=temperature)
        return self.global_info_nce(raw, augmented) + self.beta * self.local_info_nce(
            augmented, segments=self.k
        )

    # ------------------------------------------------------------------
    # Meta alternating optimization step
    # ------------------------------------------------------------------

    def meta_step(self, x, meta_opt, meta_head_opt, temperature=1.0):
        """Paper Eq. 5: jointly optimize augmentation variety and fidelity."""
        batch_size = x.size(0)
        was_training = self.encoder.training
        requires_grad = [
            parameter.requires_grad for parameter in self.encoder.parameters()
        ]
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()

        meta_opt.zero_grad(set_to_none=True)
        meta_head_opt.zero_grad(set_to_none=True)

        augmented, raw = self.get_features(x, temperature=temperature)
        labels = torch.arange(batch_size, dtype=torch.long, device=x.device)
        fidelity = F.cross_entropy(self.meta_unsup_head(augmented.amax(dim=1)), labels)
        meta_loss = self.l1out(raw, augmented) + self.meta_beta * fidelity
        meta_loss.backward()

        meta_opt.step()
        meta_head_opt.step()

        for parameter, required in zip(self.encoder.parameters(), requires_grad):
            parameter.requires_grad_(required)
        self.encoder.train(was_training)
        return meta_loss.item()

    # ------------------------------------------------------------------
    # Supervised forward pass
    # ------------------------------------------------------------------

    def representation(self, x):
        return self.encoder(x, mask="all_true").amax(dim=1)

    def forward(self, x, **kwargs):
        """Encode a sequence and apply the fitted linear forecasting head.

        Args:
            x: [B, T, D]
        Returns:
            [B, pred_len, out_ch]
        """
        out = self.head(self.representation(x))
        return out.view(x.size(0), self._pred_len, self._out_ch)
