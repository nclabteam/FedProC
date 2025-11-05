import inspect
import os
from functools import partial
from pathlib import Path

import einops
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image
from timm.models.vision_transformer import Block, PatchEmbed
from torch import nn
from torchvision.transforms import Resize
from tqdm import tqdm


def get_mae_arch():
    """Factory function to return MAE_ARCH with proper function references"""
    return {
        "mae_base": [
            mae_vit_base_patch16_dec512d8b,
            "mae_visualize_vit_base.pth",
        ],  # decoder: 512 dim, 8 blocks
        "mae_large": [
            mae_vit_large_patch16_dec512d8b,
            "mae_visualize_vit_large.pth",
        ],  # decoder: 512 dim, 8 blocks
        "mae_huge": [
            mae_vit_huge_patch14_dec512d8b,
            "mae_visualize_vit_huge.pth",
        ],  # decoder: 512 dim, 8 blocks
    }


POSSIBLE_SEASONALITIES = {
    "S": [3600],  # 1 hour
    "T": [1440, 10080],  # 1 day or 1 week
    "H": [24, 168],  # 1 day or 1 week
    "D": [7, 30, 365],  # 1 week, 1 month or 1 year
    "W": [52, 4],  # 1 year or 1 month
    "M": [12, 6, 3],  # 3 months, 6 months or 1 year
    "B": [5],
    "Q": [4, 2],  # 6 months or 1 year
}

MAE_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/mae/visualize/"

optional = {
    "arch": "mae_base",
    "finetune_type": "ln",
    "ckpt_dir": "./ckpt/",
    "load_ckpt": True,
    "context_len": 1152,
    "periodicity": 24,
    "norm_const": 0.4,
    "align_const": 0.4,
    "interpolation": "bilinear",
    "export_image": False,
    "fp64": False,
}


def args_update(parser):
    parser.add_argument(
        "--arch", type=str, default=None, choices=["mae_base", "mae_large", "mae_huge"]
    )
    parser.add_argument(
        "--finetune_type",
        type=str,
        default=None,
        choices=["full", "ln", "bias", "none", "mlp", "attn"],
    )
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--load_ckpt", type=bool, default=None)
    parser.add_argument("--context_len", type=int, default=None)
    parser.add_argument("--periodicity", type=int, default=None)
    parser.add_argument("--norm_const", type=float, default=None)
    parser.add_argument("--align_const", type=float, default=None)
    parser.add_argument(
        "--interpolation",
        type=str,
        default=None,
        choices=["bilinear", "nearest", "bicubic"],
    )
    parser.add_argument("--export_image", type=bool, default=None)
    parser.add_argument("--fp64", type=bool, default=None)


class VisionTS(nn.Module):
    """
    Paper: https://arxiv.org/abs/2408.17253
    Source: https://github.com/Keytoyze/VisionTS/blob/main/visionts/model.py
    """

    def __init__(self, configs):
        super().__init__()
        arch = configs.arch
        finetune_type = configs.finetune_type
        ckpt_dir = configs.ckpt_dir
        load_ckpt = configs.load_ckpt
        self.context_len = configs.context_len
        self.pred_len = configs.output_len
        self.periodicity = configs.periodicity
        self.norm_const = configs.norm_const
        self.export_image = configs.export_image
        self.fp64 = configs.fp64
        align_const = configs.align_const
        interpolation = configs.interpolation

        # Calculate padding
        self.pad_left = 0
        self.pad_right = 0

        MAE_ARCH = get_mae_arch()

        if arch not in MAE_ARCH:
            raise ValueError(
                f"Unknown arch: {arch}. Should be in {list(MAE_ARCH.keys())}"
            )

        # Create the vision model
        self.vision_model = MAE_ARCH[arch][0]()

        # Load checkpoint if specified
        if load_ckpt:
            ckpt_path = os.path.join(ckpt_dir, MAE_ARCH[arch][1])
            if not os.path.isfile(ckpt_path):
                remote_url = MAE_DOWNLOAD_URL + MAE_ARCH[arch][1]
                download_file(remote_url, ckpt_path)
            try:
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                self.vision_model.load_state_dict(checkpoint["model"], strict=True)
                print(f"Loaded checkpoint from {ckpt_path}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print(f"Please delete {ckpt_path} and redownload!")

        # Set parameter training based on finetune_type
        if finetune_type != "full":
            for n, param in self.vision_model.named_parameters():
                if "ln" == finetune_type:
                    param.requires_grad = "norm" in n
                elif "bias" == finetune_type:
                    param.requires_grad = "bias" in n
                elif "none" == finetune_type:
                    param.requires_grad = False
                elif "mlp" in finetune_type:
                    param.requires_grad = ".mlp." in n
                elif "attn" in finetune_type:
                    param.requires_grad = ".attn." in n

        # Model configuration
        self.image_size = self.vision_model.patch_embed.img_size[0]
        self.patch_size = self.vision_model.patch_embed.patch_size[0]
        self.num_patch = self.image_size // self.patch_size

        if self.context_len % self.periodicity != 0:
            self.pad_left = self.periodicity - self.context_len % self.periodicity

        if self.pred_len % self.periodicity != 0:
            self.pad_right = self.periodicity - self.pred_len % self.periodicity

        # Calculate patch distribution
        input_ratio = (self.pad_left + self.context_len) / (
            self.pad_left + self.context_len + self.pad_right + self.pred_len
        )
        self.num_patch_input = max(1, int(input_ratio * self.num_patch * align_const))
        self.num_patch_output = self.num_patch - self.num_patch_input
        adjust_input_ratio = self.num_patch_input / self.num_patch

        # Setup interpolation
        interpolation_mode = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]

        self.input_resize = safe_resize(
            (self.image_size, int(self.image_size * adjust_input_ratio)),
            interpolation=interpolation_mode,
        )

        self.scale_x = ((self.pad_left + self.context_len) // self.periodicity) / (
            int(self.image_size * adjust_input_ratio)
        )

        self.output_resize = safe_resize(
            (self.periodicity, int(round(self.image_size * self.scale_x))),
            interpolation=interpolation_mode,
        )

        # Create mask - fix device assignment issue
        self.num_patch_total = self.num_patch * self.num_patch
        mask = torch.ones((self.num_patch, self.num_patch), dtype=torch.float32)
        mask[:, : self.num_patch_input] = 0.0

        # Register as buffer to handle device placement automatically
        self.register_buffer("mask", mask.reshape((1, -1)))
        self.mask_ratio = torch.mean(mask).item()

    def forward(self, x):
        # Input: [batch_size, input_len, input_channels]
        # Output: [batch_size, output_len, output_channels]

        batch_size, seq_len, n_vars = x.shape

        # Validate input dimensions
        if seq_len != self.context_len:
            print(
                f"Warning: Expected sequence length {self.context_len}, got {seq_len}"
            )

        # 1. Normalization with numerical stability
        means = x.mean(1, keepdim=True).detach()  # [bs x 1 x nvars]
        x_enc = x - means

        # Use fp64 for variance calculation if specified
        x_var = x_enc.to(torch.float64) if self.fp64 else x_enc
        stdev = torch.sqrt(
            torch.var(x_var, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).to(x.dtype)
        # stdev: [bs x 1 x nvars]

        # Prevent division by zero
        stdev = torch.clamp(stdev / self.norm_const, min=1e-6)
        x_enc = x_enc / stdev

        # Channel Independent - rearrange to [bs x nvars x seq_len]
        x_enc = einops.rearrange(x_enc, "b s n -> b n s")

        # 2. Segmentation with proper padding
        x_pad = F.pad(x_enc, (self.pad_left, 0), mode="replicate")  # [b n s]

        # Ensure divisibility for reshape
        padded_len = x_pad.shape[-1]
        if padded_len % self.periodicity != 0:
            extra_pad = self.periodicity - (padded_len % self.periodicity)
            x_pad = F.pad(x_pad, (0, extra_pad), mode="replicate")

        x_2d = einops.rearrange(x_pad, "b n (p f) -> (b n) 1 f p", f=self.periodicity)

        # 3. Render & Alignment with error checking
        try:
            x_resize = self.input_resize(x_2d)
        except Exception as e:
            print(f"Error in input_resize: {e}")
            print(f"x_2d shape: {x_2d.shape}")
            raise e

        # Create masked region with proper dimensions
        masked_height = self.image_size
        masked_width = self.num_patch_output * self.patch_size

        masked = torch.zeros(
            (x_2d.shape[0], 1, masked_height, masked_width),
            device=x_2d.device,
            dtype=x_2d.dtype,
        )

        # Concatenate resized input with masked region
        try:
            x_concat_with_masked = torch.cat([x_resize, masked], dim=-1)
        except RuntimeError as e:
            print(f"Error concatenating tensors: {e}")
            print(f"x_resize shape: {x_resize.shape}")
            print(f"masked shape: {masked.shape}")
            raise e

        # Convert to 3-channel image
        image_input = einops.repeat(x_concat_with_masked, "b 1 h w -> b c h w", c=3)

        # 4. Vision model forward pass with error handling
        try:
            # Ensure mask is on the same device as input
            mask_expanded = einops.repeat(
                self.mask, "1 l -> n l", n=image_input.shape[0]
            )

            # Forward pass through vision model
            _, y, mask_output = self.vision_model(
                image_input,
                mask_ratio=self.mask_ratio,
                noise=mask_expanded,
            )

            # Reconstruct image
            image_reconstructed = self.vision_model.unpatchify(y)
            # image_reconstructed: [(bs x nvars) x 3 x h x w]

        except RuntimeError as e:
            print(f"CUDA error in vision model: {e}")
            print(f"image_input shape: {image_input.shape}")
            print(f"mask shape: {mask_expanded.shape}")
            print(f"mask_ratio: {self.mask_ratio}")

            # Clear CUDA cache and retry with smaller batch if possible
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e

        # 5. Forecasting with proper error handling
        try:
            # Convert to grayscale
            y_grey = torch.mean(image_reconstructed, 1, keepdim=True)  # color to grey

            # Resize back to original dimensions
            y_segmentations = self.output_resize(y_grey)

            # Flatten and extract forecasting window
            y_flatten = einops.rearrange(
                y_segmentations,
                "(b n) 1 f p -> b (p f) n",
                b=batch_size,
                f=self.periodicity,
            )

            # Extract the prediction window
            start_idx = self.pad_left + self.context_len
            end_idx = start_idx + self.pred_len
            y = y_flatten[:, start_idx:end_idx, :]

        except Exception as e:
            print(f"Error in forecasting step: {e}")
            print(f"image_reconstructed shape: {image_reconstructed.shape}")
            raise e

        # 6. Denormalization
        y = y * stdev.repeat(1, self.pred_len, 1)
        y = y + means.repeat(1, self.pred_len, 1)

        # Handle image export if requested
        if self.export_image:
            mask_vis = mask_output.detach()
            mask_vis = mask_vis.unsqueeze(-1).repeat(
                1, 1, self.vision_model.patch_embed.patch_size[0] ** 2 * 3
            )
            mask_vis = self.vision_model.unpatchify(mask_vis)
            image_reconstructed_vis = (
                image_input * (1 - mask_vis) + image_reconstructed * mask_vis
            )
            green_bg = -torch.ones_like(image_reconstructed) * 2
            image_input_vis = image_input * (1 - mask_vis) + green_bg * mask_vis

            image_input_vis = einops.rearrange(
                image_input_vis, "(b n) c h w -> b n h w c", b=batch_size
            )
            image_reconstructed_vis = einops.rearrange(
                image_reconstructed_vis, "(b n) c h w -> b n h w c", b=batch_size
            )
            return y, image_input_vis, image_reconstructed_vis

        return y


# Rest of the model definitions remain the same...
class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        quantile=False,
        quantile_head_num=9,
    ):
        super().__init__()

        # add for quantile forecasting
        self.norm_pix_loss = norm_pix_loss
        self.quantile = quantile
        self.quantile_head_num = quantile_head_num

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # prediction head
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch
        if self.quantile:
            # adds quantile outputs:
            # we need 9 outputs in total, corresponding to 10%, 20%, ..., 90%
            # So other 8 heads are needed to generate other quantile outputs.
            self.decoder_pred_quantile_list = nn.ModuleList()
            for i in range(self.quantile_head_num - 1):
                self.decoder_pred_quantile_list.append(
                    nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
                )  # decoder to patch

        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 * 3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x, n_channels=3):
        """
        x: (N, L, patch_size**2 * n_channels)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, n_channels))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], n_channels, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        noise: [N, L]
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(round(L * (1 - mask_ratio)))

        if noise is None:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, noise=None):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio, noise)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        if not self.quantile:
            x = self.decoder_pred(x)
            x = x[:, 1:, :]  # remove cls token
            return x
        else:
            # first calculate the 50% quantile value
            x_mid = self.decoder_pred(x)[:, 1:, :]  # [batch, ]

            # then calculate the other quantile values
            x_quantile_list = []
            for i in range(self.quantile_head_num - 1):
                x_quantile = self.decoder_pred_quantile_list[i](x)[:, 1:, :]
                x_quantile_list.append(x_quantile)

            # this will generate 9 groups of data
            return x_mid, x_quantile_list

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75, noise=None):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, noise)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        return None, pred, mask


# Model factory functions
def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# Utility functions remain the same...
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


def download_file(url, local_filename):
    response = requests.get(url, stream=True)
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(local_filename, "wb") as file:
        with tqdm(
            desc=f"Download: {local_filename}",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))


def safe_resize(size, interpolation):
    signature = inspect.signature(Resize)
    params = signature.parameters
    if "antialias" in params:
        return Resize(size, interpolation, antialias=False)
    else:
        return Resize(size, interpolation)


def norm_freq_str(freq_str: str) -> str:
    base_freq = freq_str.split("-")[0]
    if len(base_freq) >= 2 and base_freq.endswith("S"):
        return base_freq[:-1]
    return base_freq


def freq_to_seasonality_list(freq: str, mapping_dict=None) -> int:
    if mapping_dict is None:
        mapping_dict = POSSIBLE_SEASONALITIES
    offset = pd.tseries.frequencies.to_offset(freq)
    base_seasonality_list = mapping_dict.get(norm_freq_str(offset.name), [])
    seasonality_list = []
    for base_seasonality in base_seasonality_list:
        seasonality, remainder = divmod(base_seasonality, offset.n)
        if not remainder:
            seasonality_list.append(seasonality)

    # Append P=1 for those without significant periodicity
    seasonality_list.append(1)
    return seasonality_list
