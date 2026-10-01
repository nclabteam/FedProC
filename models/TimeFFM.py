from math import sqrt

import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

from models import CHECKPOINT_DIR


class TimeFFMReprogramming(nn.Module):
    """Prompt adaption: select top-M text prototypes per head (paper Sec. 4.2)."""

    def __init__(self, d_model: int, n_heads: int, topk: int) -> None:
        super().__init__()
        d_keys = d_model // n_heads
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        self.dropout = nn.Dropout(0.1)
        self.n_heads = n_heads
        self.topk = topk

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        B, L, _ = target.shape
        S, _ = source.shape
        H = self.n_heads
        q = self.query_projection(target).view(B, L, H, -1)
        k = self.key_projection(source).view(S, H, -1)
        scores = torch.einsum("blhe,she->bhls", q, k)
        A = self.dropout(torch.softmax(scores / sqrt(q.shape[-1]), dim=-1))
        # TopM over attention mass summed across patches, per head
        idxs = torch.topk(A.sum(dim=2), self.topk).indices  # B, H, M
        k = k.permute(1, 0, 2)  # H, S, E
        z = k[torch.arange(H)[None, :, None], idxs]  # B, H, M, E
        z = self.dropout(z.permute(0, 2, 1, 3)).reshape(B, self.topk, -1)
        return self.out_projection(z)


class TimeFFMEncoder(nn.Module):
    """Modality alignment + prompt adaption: the federated (global) part."""

    def __init__(self, configs, word_embeddings: torch.Tensor) -> None:
        super().__init__()
        d_model = word_embeddings.shape[1]
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.feature_embedding = nn.Linear(configs.patch_len, d_model)
        self.ts_embed_dropout = nn.Dropout(configs.ts_embed_dropout)
        # Frozen copy of GPT-2 token embeddings; never trained or transmitted.
        self.register_buffer("word_embeddings", word_embeddings, persistent=False)
        self.mapping_layer = nn.Linear(word_embeddings.shape[0], configs.num_tokens)
        self.prompt_embedding = TimeFFMReprogramming(
            d_model=d_model, n_heads=configs.n_heads, topk=configs.topk
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, C, L (already normalized)
        L = x.shape[-1]
        if L <= self.patch_len:
            pad = self.patch_len - L
        elif L % self.stride == 0:
            pad = 0
        else:
            pad = (L // self.stride) * self.stride + self.patch_len - L
        x = nn.functional.pad(x, (0, pad), mode="replicate")
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        b, c, p, h = x.shape
        tokens = self.ts_embed_dropout(self.feature_embedding(x.reshape(b * c, p, h)))
        source = self.mapping_layer(self.word_embeddings.T).T  # num_tokens, D
        prompts = self.prompt_embedding(tokens, source)
        return torch.cat((prompts, tokens), dim=1)


class TimeFFM(nn.Module):
    """Time-FFM (Liu et al., NeurIPS 2024): encoder -> frozen GPT-2 -> personal head.

    The head predicts the backcast and the forecast (official code); ``forward``
    returns only the forecast, ``forward_full`` returns both for training.
    """

    optional = {
        "lm_layer_num": 6,
        "patch_len": 16,
        "stride": 16,
        "num_tokens": 100,
        "topk": 12,
        "n_heads": 8,
        "ts_embed_dropout": 0.3,
        "dec_head_dropout": 0.1,
    }

    @classmethod
    def args_update(cls, parser) -> None:
        parser.add_argument("--lm_layer_num", type=int, default=None)
        parser.add_argument("--patch_len", type=int, default=None)
        parser.add_argument("--stride", type=int, default=None)
        parser.add_argument("--num_tokens", type=int, default=None)
        parser.add_argument("--topk", type=int, default=None)
        parser.add_argument("--n_heads", type=int, default=None)
        parser.add_argument("--ts_embed_dropout", type=float, default=None)
        parser.add_argument("--dec_head_dropout", type=float, default=None)

    def __init__(self, configs) -> None:
        super().__init__()
        self.input_len = configs.input_len
        self.output_len = configs.output_len
        self.gpt2 = GPT2Model.from_pretrained("gpt2", cache_dir=str(CHECKPOINT_DIR))
        self.gpt2.h = self.gpt2.h[: configs.lm_layer_num]
        for param in self.gpt2.parameters():
            param.requires_grad = False
        self.encoder = TimeFFMEncoder(
            configs=configs,
            word_embeddings=self.gpt2.get_input_embeddings().weight.detach().clone(),
        )
        with torch.no_grad():
            n_tokens = self.encoder(torch.zeros(1, 1, configs.input_len)).shape[1]
        d_model = self.gpt2.config.n_embd
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(d_model * n_tokens, configs.input_len + configs.output_len),
            nn.Dropout(configs.dec_head_dropout),
        )

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """x: B, L, C -> B, L + H, C (backcast + forecast)."""
        B, _, C = x.shape
        means = x.mean(dim=1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-5).detach()
        x = x / stdev
        embeds = self.encoder(x.transpose(1, 2))
        hidden = self.gpt2(inputs_embeds=embeds).last_hidden_state  # B*C, T, D
        hidden = hidden.reshape(B, C, *hidden.shape[-2:]).permute(0, 1, 3, 2)
        out = self.head(hidden).transpose(1, 2)  # B, L + H, C
        return out * stdev + means

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward_full(x)[:, -self.output_len :]
