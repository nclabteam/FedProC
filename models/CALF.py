import torch
import torch.nn as nn
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model
from layers.AccustumGPT2 import AccustumGPT2Model, gpt2_pca_embeddings


class CALF(nn.Module):

    optional = {
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "r": 8,
        "gpt_layers": 6,
        "d_model": 768,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--lora_alpha", type=int, default=None)
        parser.add_argument("--lora_dropout", type=float, default=None)
        parser.add_argument("--r", type=int, default=None)
        parser.add_argument("--gpt_layers", type=int, default=None)
        parser.add_argument("--d_model", type=int, default=None)

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.output_len

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn"],
            fan_in_fan_out=True,
        )

        # Load pretrained GPT-2 models with error handling and caching
        try:
            self.gpt2 = AccustumGPT2Model.from_pretrained(
                "gpt2",
                output_attentions=True,
                output_hidden_states=True,
                local_files_only=False,
                cache_dir=None,
            )
            self.gpt2_text = AccustumGPT2Model.from_pretrained(
                "gpt2",
                output_attentions=True,
                output_hidden_states=True,
                local_files_only=False,
                cache_dir=None,
            )
        except Exception as e:
            raise RuntimeError(f"Error loading GPT-2 models: {e}")

        # Trim to specified number of layers
        self.gpt2.h = self.gpt2.h[: configs.gpt_layers]
        self.gpt2_text.h = self.gpt2_text.h[: configs.gpt_layers]

        # Apply PEFT to the main GPT-2 model
        self.gpt2 = get_peft_model(self.gpt2, peft_config)

        # Create word embeddings directly in memory
        word_embedding = self._create_pca_embeddings(configs)

        # Set parameter training flags
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if "ln" in name or "wpe" in name or "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for i, (name, param) in enumerate(self.gpt2_text.named_parameters()):
            if "wpe" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Initialize projection layers
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

        # Input and output layers
        self.in_layer = Encoder_PCA(
            input_dim=configs.input_len,
            word_embedding=word_embedding,
            hidden_dim=configs.d_model,
        )

        self.out_layer = nn.Linear(configs.d_model, configs.output_len)

        for layer in (
            self.gpt2_text,
            self.gpt2,
            self.in_layer,
            self.out_layer,
            self.time_proj,
            self.text_proj,
        ):
            layer.train()

        self.cnt = 0

    def _create_pca_embeddings(self, configs):
        try:
            return gpt2_pca_embeddings(self.gpt2_text)
        except Exception:
            return torch.randn(configs.d_model, 500)

    def forward(self, x, **kwargs):
        # x: [B, L, M] where B=batch, L=seq_len, M=features
        B, L, M = x.shape

        # Normalize input
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x /= stdev

        # Rearrange for processing: [B, M, L]
        x = rearrange(x, "b l m -> b m l")

        # Encode input
        outputs_time1, outputs_text1 = self.in_layer(x)

        # Process through GPT models
        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(
            inputs_embeds=outputs_text1
        )

        # Residual connections
        outputs_time += outputs_time1
        outputs_text += outputs_text1

        # Apply projections to intermediate features
        intermidiate_feat_time = tuple(
            [
                self.time_proj[idx](feat)
                for idx, feat in enumerate(list(intermidiate_feat_time))
            ]
        )
        intermidiate_feat_text = tuple(
            [
                self.text_proj[idx](feat)
                for idx, feat in enumerate(list(intermidiate_feat_text))
            ]
        )

        # Generate outputs
        outputs_time = self.out_layer(outputs_time[:, -M:, :])
        outputs_text = self.out_layer(outputs_text[:, -M:, :])

        # Rearrange back to [B, L, M]
        outputs_time = rearrange(outputs_time, "b m l -> b l m")
        outputs_text = rearrange(outputs_text, "b m l -> b l m")

        # Denormalize
        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        output = {
            "outputs_text": outputs_text,
            "outputs_time": outputs_time,
            "intermidiate_time": intermidiate_feat_time,
            "intermidiate_text": intermidiate_feat_text,
        }

        return outputs_time


class Encoder_PCA(nn.Module):
    def __init__(
        self,
        input_dim,
        word_embedding,
        hidden_dim=768,
        num_heads=12,
        num_encoder_layers=1,
    ):
        super(Encoder_PCA, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                batch_first=True,
            ),
            num_layers=num_encoder_layers,
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads
        )

        self.word_embedding = word_embedding.T

    def forward(self, x, **kwargs):
        B = x.shape[0]

        self.word_embedding = self.word_embedding.to(x.device)

        if self.word_embedding.ndim == 2:
            self.word_embedding = self.word_embedding.repeat(B, 1, 1)
        elif self.word_embedding.shape[0] != B:
            self.word_embedding = self.word_embedding[0].repeat(B, 1, 1)

        x = self.linear(x)
        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)
        x_time = x

        # Cross attention with word embeddings
        q = x.transpose(0, 1)
        k = v = self.word_embedding.transpose(0, 1)
        x, _ = self.cross_attention(q, k, v)
        x = x.transpose(0, 1)

        return x_time, x
