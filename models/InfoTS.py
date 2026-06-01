import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from augs import (cutout, jitter, magnitude_warp, scaling, subsequence,
                  time_warp, window_slice, window_warp)

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


# ------------------------------------------------------------------
# TS Encoder Wrapper
# ------------------------------------------------------------------


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            res[i, t : t + l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


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
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "continuous":
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
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
    def __init__(self, aug_p1=0.2, aug_p2=0.0):
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
        self.weight = nn.Parameter(torch.empty(2, len(self.augs)))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)
        self.aug_p1 = aug_p1
        self.aug_p2 = aug_p2

    def get_sampling(self, temperature=1.0, bias=0.0):
        if self.training:
            bias = bias + 0.0001
            eps = (bias - (1 - bias)) * torch.rand(
                self.weight.size(), device=self.weight.device
            ) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + self.weight) / temperature
            return torch.softmax(gate_inputs, -1)
        else:
            return torch.softmax(self.weight, -1)

    def forward(self, x, temperature=1.0):
        # x: [B, T, D]
        if self.aug_p1 == 0.0 and self.aug_p2 == 0.0:
            return x.clone(), x.clone()

        para = self.get_sampling(temperature=temperature)

        if random.random() > self.aug_p1 and self.training:
            aug1 = x.clone()
        else:
            xs1_list = []
            for aug in self.augs:
                xs1_list.append(aug(x))
            xs1 = torch.stack(xs1_list, 0)  # [num_augs, B, T, D]
            xs1_flat = torch.reshape(xs1, (xs1.shape[0], -1))  # [num_augs, B * T * D]
            aug1_flat = torch.unsqueeze(para[0], -1) * xs1_flat  # [num_augs, B * T * D]
            aug1 = torch.reshape(aug1_flat, xs1.shape)
            aug1 = torch.sum(aug1, dim=0)  # [B, T, D]

        aug2 = x.clone()
        return aug1, aug2


# ------------------------------------------------------------------
# Loss Functions
# ------------------------------------------------------------------


def InfoNCE(z1, z2, temperature=1.0):
    batch_size = z1.size(0)
    features = torch.cat([z1, z2], dim=0).squeeze(1)  # 2B x C

    labels = torch.cat(
        [torch.arange(batch_size, device=z1.device) for _ in range(2)], dim=0
    )
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=z1.device)
    labels = labels[~mask].view(labels.shape[0], -1)

    similarity_matrix = torch.matmul(features, features.T)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits = logits / temperature
    logits = -F.log_softmax(logits, dim=-1)
    return logits[:, 0].mean()


def global_infoNCE(z1, z2, pooling="max", temperature=1.0):
    if pooling == "max":
        z1 = F.max_pool1d(
            z1.transpose(1, 2).contiguous(), kernel_size=z1.size(1)
        ).transpose(1, 2)
        z2 = F.max_pool1d(
            z2.transpose(1, 2).contiguous(), kernel_size=z2.size(1)
        ).transpose(1, 2)
    elif pooling == "mean":
        z1 = torch.unsqueeze(torch.mean(z1, 1), 1)
        z2 = torch.unsqueeze(torch.mean(z2, 1), 1)

    return InfoNCE(z1, z2, temperature)


def local_infoNCE(z1, z2, pooling="max", temperature=1.0, k=8):
    B, T, D = z1.size()
    crop_size = int(T / k)
    crop_len = crop_size * k
    if crop_len <= 0 or T < crop_len:
        return torch.tensor(0.0, device=z1.device)

    start = random.randint(0, T - crop_len)
    crop_z1 = z1[:, start : start + crop_len, :]
    crop_z1 = crop_z1.view(B, k, crop_size, D)

    if pooling == "max":
        crop_z1 = crop_z1.reshape(B * k, crop_size, D)
        crop_z1_pooling = (
            F.max_pool1d(crop_z1.transpose(1, 2).contiguous(), kernel_size=crop_size)
            .transpose(1, 2)
            .reshape(B, k, D)
        )
    elif pooling == "mean":
        crop_z1_pooling = torch.unsqueeze(torch.mean(z1, 1), 1)

    crop_z1_pooling_T = crop_z1_pooling.transpose(1, 2)
    similarity_matrices = torch.bmm(crop_z1_pooling, crop_z1_pooling_T)

    labels = torch.eye(k - 1, dtype=torch.float32, device=z1.device)
    labels = torch.cat([labels, torch.zeros(1, k - 1, device=z1.device)], 0)
    labels = torch.cat([torch.zeros(k, 1, device=z1.device), labels], -1)

    pos_labels = labels.clone()
    pos_labels[k - 1, k - 2] = 1.0

    neg_labels = labels.T + labels + torch.eye(k, device=z1.device)
    neg_labels[0, 2] = 1.0
    neg_labels[-1, -3] = 1.0

    similarity_matrix = similarity_matrices[0]
    positives = similarity_matrix[pos_labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~neg_labels.bool()].view(
        similarity_matrix.shape[0], -1
    )

    logits = torch.cat([positives, negatives], dim=1)
    logits = logits / temperature
    logits = -F.log_softmax(logits, dim=-1)
    return logits[:, 0].mean()


# ------------------------------------------------------------------
# Main InfoTS Module
# ------------------------------------------------------------------


class InfoTS(nn.Module):
    optional = {
        "infots_repr_dim": 320,
        "infots_hidden_dim": 64,
        "infots_depth": 10,
        "infots_beta": 1.0,
        "infots_meta_beta": 1.0,
        "infots_aug_p1": 0.2,
        "infots_aug_p2": 0.0,
        "infots_k": 8,
        "infots_meta_lr": 0.01,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--infots_repr_dim", type=int, default=None)
        parser.add_argument("--infots_hidden_dim", type=int, default=None)
        parser.add_argument("--infots_depth", type=int, default=None)
        parser.add_argument("--infots_beta", type=float, default=None)
        parser.add_argument("--infots_meta_beta", type=float, default=None)
        parser.add_argument("--infots_aug_p1", type=float, default=None)
        parser.add_argument("--infots_aug_p2", type=float, default=None)
        parser.add_argument("--infots_k", type=int, default=None)
        parser.add_argument("--infots_meta_lr", type=float, default=None)

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

        # Differentiable learnable Auto-Augmentation
        self.aug = AutoAUG(aug_p1=configs.infots_aug_p1, aug_p2=configs.infots_aug_p2)

        # Supervised mapping head for forecasting
        self.head = nn.Linear(repr_dim, pred_len * out_ch)
        self._pred_len = pred_len
        self._out_ch = out_ch

        # Unsupervised/Supervised classifier head used for the Meta-Update
        self.meta_unsup_head = nn.Linear(repr_dim, configs.batch_size)
        self.CE = nn.CrossEntropyLoss()

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
        out1, out2 = self.get_features(x, temperature=temperature)
        loss = (
            global_infoNCE(out1, out2) + local_infoNCE(out1, out2, k=self.k) * self.beta
        )
        return loss

    # ------------------------------------------------------------------
    # Meta alternating optimization step
    # ------------------------------------------------------------------

    def meta_step(self, x, meta_opt, meta_head_opt, temperature=1.0):
        """Performs one step of AutoAUG weight tuning to optimize representation variety."""
        B = x.size(0)
        self.encoder.eval()

        meta_opt.zero_grad(set_to_none=True)
        meta_head_opt.zero_grad(set_to_none=True)

        outv, outx = self.get_features(x, temperature=temperature)

        # Pick random target class indexes for batch validation
        y = torch.arange(B, dtype=torch.long, device=x.device)

        zv = F.max_pool1d(
            outv.transpose(1, 2).contiguous(), kernel_size=outv.size(1)
        ).squeeze(2)
        zx = F.max_pool1d(
            outx.transpose(1, 2).contiguous(), kernel_size=outx.size(1)
        ).squeeze(2)

        pred_yv = self.meta_unsup_head(zv)
        pred_yx = self.meta_unsup_head(zx)

        # Meta Variety / Fidelity trade-off objective
        loss_vy = self.CE(pred_yv, y)
        loss_xy = self.CE(pred_yx, y)

        meta_loss = self.meta_beta * (loss_vy + loss_xy)
        meta_loss.backward()

        meta_opt.step()
        meta_head_opt.step()

        self.encoder.train()
        return meta_loss.item()

    # ------------------------------------------------------------------
    # Supervised forward pass
    # ------------------------------------------------------------------

    def forward(self, x, **kwargs):
        """Standard casual encoding + projection for time-series forecasting.

        Args:
            x: [B, T, D]
        Returns:
            [B, pred_len, out_ch]
        """
        # Encode sequence
        repr_seq = self.encoder(x, mask="all_true")  # [B, T, repr_dim]
        # Max pool over sequence length to get sequence representation summary
        repr_sum = F.max_pool1d(
            repr_seq.transpose(1, 2).contiguous(), kernel_size=repr_seq.size(1)
        ).squeeze(
            2
        )  # [B, repr_dim]
        out = self.head(repr_sum)  # [B, pred_len * out_ch]
        return out.view(x.size(0), self._pred_len, self._out_ch)
