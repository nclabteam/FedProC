import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model
from transformers import GPT2Tokenizer

from layers.AccustumGPT2 import AccustumGPT2Model, gpt2_pca_embeddings
from layers.RevIN import RevIN

# ---------------------------------------------------------------------------
# Inception blocks (used only by LLM_TPF via Freq_Block)
# ---------------------------------------------------------------------------


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernels = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
                for i in range(num_kernels)
            ]
        )
        if init_weight:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return torch.stack([k(x) for k in self.kernels], dim=-1).mean(-1)


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super().__init__()
        self.num_kernels = num_kernels
        kernels = []
        for i in range(num_kernels // 2):
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=[1, 2 * i + 3],
                    padding=[0, i + 1],
                )
            )
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=[2 * i + 3, 1],
                    padding=[i + 1, 0],
                )
            )
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return torch.stack([k(x) for k in self.kernels], dim=-1).mean(-1)


# ---------------------------------------------------------------------------
# Frequency block (used only by LLM_TPF)
# ---------------------------------------------------------------------------


def _random_select_with_temperature(frequency_list, k, temperature=0.9):
    freq = torch.tensor(frequency_list, dtype=torch.float32)
    probs = torch.softmax(freq / temperature, dim=0)
    idx = torch.multinomial(probs, k, replacement=False)
    return freq[idx], idx


def _fft_for_period(x, weights, k=2):
    xf = torch.fft.rfft(x, dim=1)
    normalized_weights = weights / weights.sum()
    frequency_list = (abs(xf) * normalized_weights).mean(0).sum(-1)
    frequency_list[0] = 0
    _, top_list = _random_select_with_temperature(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    weighted_mean = (abs(xf) * normalized_weights).sum(-1)[:, top_list]
    return period, weighted_mean


class Freq_Block(nn.Module):
    def __init__(self, configs, device, word_embedding):
        super().__init__()
        self.weights = None
        self.configs = configs
        self.device = device
        self.seq_len = configs.seq_len
        self.k = configs.timesnet_k
        self.word_embedding = word_embedding.T
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=12)

    def forward(self, x):
        B, T, N = x.size()
        if self.word_embedding.ndim == 2:
            self.word_embedding = self.word_embedding.repeat(B, 1, 1)
        elif self.word_embedding.shape[0] != B:
            self.word_embedding = self.word_embedding[0].repeat(B, 1, 1)
        out_channel = 32
        conv = nn.Sequential(
            Inception_Block_V2(N, out_channel, num_kernels=self.configs.num_kernels),
            nn.GELU(),
            Inception_Block_V2(out_channel, N, num_kernels=self.configs.num_kernels),
        ).to(self.device)
        weights = nn.Parameter(torch.ones(x.shape[-1])).to(self.device)
        period_list, period_weight = _fft_for_period(x, weights, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if period == 0:
                period = 1
            if self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros([B, length - self.seq_len, N]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            out = (
                out.reshape(B, length // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            out = conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            out = out[:, : self.seq_len, :]
            linear1 = nn.Linear(out.shape[-1], 768).to(self.device)
            linear2 = nn.Linear(768, out.shape[-1]).to(self.device)
            out = linear1(out.transpose(0, 1))
            out, _ = self.cross_attention(
                out,
                self.word_embedding.transpose(0, 1),
                self.word_embedding.transpose(0, 1),
            )
            out = linear2(out.transpose(0, 1))
            res.append(out)
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = x  # residual connection
        return res


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class Encoder_PCA(nn.Module):
    def __init__(
        self,
        input_dim,
        word_embedding,
        hidden_dim=768,
        num_heads=12,
        num_encoder_layers=1,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads
        )
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1)
        self.word_embedding = word_embedding.T

    def forward(self, x, prompt, time_fusion):
        B = x.shape[0]
        if self.word_embedding.ndim == 2:
            self.word_embedding = self.word_embedding.repeat(B, 1, 1)
        elif self.word_embedding.shape[0] != B:
            self.word_embedding = self.word_embedding[0].repeat(B, 1, 1)
        x = self.linear(x)
        time_fusion = self.linear(time_fusion)
        x_time = x
        time_pub = torch.cat((prompt, x_time), dim=1)
        time_private_prompt = time_pub
        time_pub, _ = self.self_attention(time_pub, time_pub, time_pub)
        q = time_fusion.transpose(0, 1)
        k = v = time_pub.transpose(0, 1)
        time_pub, _ = self.cross_attention(q, k, v)
        time_pub = time_pub.transpose(0, 1)
        time_private_fusion = time_fusion
        time_private_prompt = time_private_prompt.transpose(1, 2)
        time_private_prompt = nn.Linear(
            time_private_prompt.shape[-1], time_private_fusion.shape[1]
        ).to(self.linear.weight.device)(time_private_prompt)
        time_private_prompt = time_private_prompt.transpose(1, 2)
        return time_pub, time_private_fusion, time_private_prompt


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _prompt_build(x, description, factor, seq_len, pred_len):
    B = x.shape[0]
    return [
        (
            f"<|start_prompt|>Dataset description: {description} "
            f"<|start_prompt|>External factors: {factor} "
            f"Task description: forecast the next {pred_len} steps given the previous {seq_len} steps information. "
        )
        for _ in range(B)
    ]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class LLM_TPF(nn.Module):

    optional = {
        "gpt_layers": 3,
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "d_model": 768,
        "num_kernels": 3,
        "timesnet_k": 3,
        "content": "time series forecasting",
        "factor": "none",
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--gpt_layers", type=int, default=None)
        parser.add_argument("--r", type=int, default=None)
        parser.add_argument("--lora_alpha", type=int, default=None)
        parser.add_argument("--lora_dropout", type=float, default=None)
        parser.add_argument("--d_model", type=int, default=None)
        parser.add_argument("--num_kernels", type=int, default=None)
        parser.add_argument("--timesnet_k", type=int, default=None)
        parser.add_argument("--content", type=str, default=None)
        parser.add_argument("--factor", type=str, default=None)

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.output_len
        self.seq_len = configs.input_len
        self.content = getattr(configs, "content", "time series forecasting")
        self.device = configs.device
        self.factor = getattr(configs, "factor", "none")
        self.revin_layer = RevIN(configs.input_channels, affine=False)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn", "c_proj"],
        )

        self.gpt2 = AccustumGPT2Model.from_pretrained(
            "gpt2", output_attentions=True, output_hidden_states=True
        )
        self.gpt2_fussion = AccustumGPT2Model.from_pretrained(
            "gpt2", output_attentions=True, output_hidden_states=True
        )
        self.gpt2_prompt = AccustumGPT2Model.from_pretrained(
            "gpt2", output_attentions=True, output_hidden_states=True
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.gpt2.h = self.gpt2.h[: configs.gpt_layers]
        self.gpt2_fussion.h = self.gpt2_fussion.h[: configs.gpt_layers]
        self.gpt2_prompt.h = self.gpt2_prompt.h[: configs.gpt_layers]

        self.gpt2 = get_peft_model(self.gpt2, peft_config)

        word_embedding = self._create_pca_embeddings()

        for name, param in self.gpt2.named_parameters():
            if "ln" in name or "wpe" in name or "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for name, param in self.gpt2_fussion.named_parameters():
            if "wpe" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.time_proj = nn.ModuleList(
            [
                nn.Linear(configs.d_model, configs.d_model, bias=False)
                for _ in range(configs.gpt_layers + 1)
            ]
        )
        self.text_proj = nn.ModuleList(
            [
                nn.Linear(configs.d_model, configs.d_model, bias=False)
                for _ in range(configs.gpt_layers + 1)
            ]
        )

        self.in_layer = Encoder_PCA(
            self.seq_len, word_embedding, hidden_dim=configs.d_model
        )
        self.out_layer = nn.Linear(configs.d_model, self.pred_len)

        for layer in (
            self.gpt2_fussion,
            self.gpt2,
            self.in_layer,
            self.out_layer,
            self.time_proj,
            self.text_proj,
        ):
            layer.to(device=self.device)
            layer.train()

        if not hasattr(configs, "seq_len"):
            configs.seq_len = self.seq_len
        self.freq_block = Freq_Block(configs, self.device, word_embedding).to(
            self.device
        )

    def _create_pca_embeddings(self):
        return gpt2_pca_embeddings(self.gpt2_prompt, device=self.device)

    def forward(self, x, **kwargs):
        B, L, M = x.shape

        x = self.revin_layer(x, "norm")

        prompt = _prompt_build(
            x, self.content, self.factor, self.seq_len, self.pred_len
        )
        prompt = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).input_ids
        prompt_embeddings = self.gpt2.get_input_embeddings()(prompt.to(x.device))

        time_fusion = self.freq_block(x)

        x = rearrange(x, "b l m -> b m l")
        time_fusion = rearrange(time_fusion, "b l m -> b m l")

        time_pub, time_private_fusion, time_private_prompt = self.in_layer(
            x, prompt=prompt_embeddings, time_fusion=time_fusion
        )

        outputs_prompt, _ = self.gpt2_prompt(inputs_embeds=time_private_prompt)
        outputs_text, _ = self.gpt2(inputs_embeds=time_pub)
        outputs_time, _ = self.gpt2_fussion(inputs_embeds=time_private_fusion)

        outputs_prompt += time_private_prompt
        outputs_text += time_pub
        outputs_time += time_private_fusion

        outputs_prompt = self.out_layer(outputs_prompt[:, -M:, :])
        outputs_text = self.out_layer(outputs_text[:, -M:, :])
        outputs_time = self.out_layer(outputs_time[:, -M:, :])

        outputs_prompt = rearrange(outputs_prompt, "b m l -> b l m")
        outputs_text = rearrange(outputs_text, "b m l -> b l m")
        outputs_time = rearrange(outputs_time, "b m l -> b l m")

        outputs_prompt = self.revin_layer(outputs_prompt, "denorm")
        outputs_time = self.revin_layer(outputs_time, "denorm")
        outputs_text = self.revin_layer(outputs_text, "denorm")

        w = nn.Parameter(torch.ones(3, device=x.device))
        w = w / w.sum()
        return w[0] * outputs_time + w[1] * outputs_text + w[2] * outputs_prompt
