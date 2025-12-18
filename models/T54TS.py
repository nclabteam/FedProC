import torch
import torch.nn as nn
from einops import rearrange
from transformers import T5Config, T5ForConditionalGeneration

optional = {
    "is_gpt": True,
    "patch_size": 16,
    "pretrain": True,
    "stride": 8,
    "gpt_layers": 6,
    "d_model": 768,
    "freeze": True,
}


def args_update(parser):
    parser.add_argument("--is_gpt", type=bool, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--pretrain", type=bool, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--gpt_layers", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--freeze", type=bool, default=None)


class T54TS(nn.Module):
    """
    Paper: https://arxiv.org/pdf/2310.04948
    Source: https://github.com/DC-research/TEMPO/blob/main/tempo/models/T5.py
    """

    def __init__(self, configs):
        super().__init__()
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.is_gpt = configs.is_gpt

        # Calculate patch_num based on input_len
        self.padding_patch_layer = nn.ReplicationPad1d((0, configs.stride))
        patch_num = (configs.input_len - configs.patch_size) // configs.stride + 1
        patch_num += 1  # Account for padding

        if configs.is_gpt:
            if configs.pretrain:
                self.t5 = T5ForConditionalGeneration.from_pretrained("t5-base")
            else:
                self.t5 = T5ForConditionalGeneration(T5Config())

            # Trim to specified number of layers
            self.t5.encoder.block = self.t5.encoder.block[: configs.gpt_layers]

        # Input projection: patch_size -> d_model
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)

        # Output projection: (d_model * patch_num) -> output_len
        self.out_layer = nn.Linear(configs.d_model * patch_num, configs.output_len)

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.t5.named_parameters()):
                if "layer_norm" in name or "relative_attention_bias" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.t5, self.in_layer, self.out_layer):
            layer.train()

    def forward(self, x):
        # Input: [batch_size, input_len, input_channels]
        B, L, M = x.shape  # B=batch_size, L=input_len, M=input_channels

        # Normalization along time dimension
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        var = torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        stdev = torch.sqrt(var).detach()
        x /= stdev

        # Rearrange for patching: [B, L, M] -> [B, M, L]
        x = rearrange(x, "b l m -> b m l")

        # Apply padding: [B, M, L] -> [B, M, L + stride]
        x = self.padding_patch_layer(x)

        # Create patches: [B, M, L + stride] -> [B, M, patch_num, patch_size]
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)

        # Rearrange: [B, M, patch_num, patch_size] -> [B*M, patch_num, patch_size]
        x = rearrange(x, "b m n p -> (b m) n p")

        # Project patches to d_model: [B*M, patch_num, patch_size] -> [B*M, patch_num, d_model]
        outputs = self.in_layer(x)

        if self.is_gpt:
            # T5 encoder processing: [B*M, patch_num, d_model] -> [B*M, patch_num, d_model]
            encoder_outputs = self.t5.encoder(inputs_embeds=outputs)
            outputs = encoder_outputs.last_hidden_state

        # Flatten for output projection: [B*M, patch_num, d_model] -> [B*M, patch_num * d_model]
        outputs = outputs.reshape(B * M, -1)

        # Project to output length: [B*M, patch_num * d_model] -> [B*M, output_len]
        outputs = self.out_layer(outputs)

        # Rearrange to final shape: [B*M, output_len] -> [B, output_len, M]
        outputs = rearrange(outputs, "(b m) l -> b l m", b=B)

        # Denormalization
        outputs = outputs * stdev + means

        # Output: [batch_size, output_len, output_channels]
        return outputs
