import os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

optional = {
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "r": 8,
    "gpt_layers": 6,
    "d_model": 768,
}


def args_update(parser):
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--r", type=int, default=None)
    parser.add_argument("--gpt_layers", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)


class CALF(nn.Module):
    """
    Paper: https://arxiv.org/abs/2403.07300
    Source: https://github.com/Hank0626/CALF/blob/main/models/CALF.py
    """

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

        # Move all components to device
        if hasattr(configs, "device"):
            device = configs.device
            for layer in (
                self.gpt2_text,
                self.gpt2,
                self.in_layer,
                self.out_layer,
                self.time_proj,
                self.text_proj,
            ):
                layer.to(device=device)
                layer.train()

        self.cnt = 0

    def _create_pca_embeddings(self, configs):
        """Create PCA-reduced word embeddings from GPT-2 directly in memory"""
        try:
            # Extract word token embeddings from the loaded model
            wte = self.gpt2_text.wte.state_dict()["weight"].cpu().numpy()

            # Apply PCA to reduce dimensions
            try:
                from sklearn.decomposition import PCA

                pca = PCA(n_components=500)
                wte_pca = pca.fit_transform(
                    wte.T
                )  # Transpose: [768, vocab_size] -> [500, vocab_size]
            except ImportError:
                # Fallback to random embeddings if sklearn not available
                wte_pca = torch.randn(500, wte.shape[0]).numpy()

            # Convert to tensor and move to device
            device = getattr(configs, "device", "cpu")
            word_embedding = torch.tensor(wte_pca).to(device=device)

            return word_embedding

        except Exception as e:
            # Fallback to random embeddings
            device = getattr(configs, "device", "cpu")
            return torch.randn(configs.d_model, 500).to(device=device)

    def forward(self, x):
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

    def forward(self, x):
        B = x.shape[0]
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


class AccustumGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        """
        Add the missing prepare_inputs_for_generation method that PEFT expects
        """
        token_type_ids = kwargs.get("token_type_ids", None)

        # Only use the last token if past_key_values is provided
        if past_key_values:
            input_ids = input_ids[:, -1:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1:]

        # Create attention mask if not provided
        if attention_mask is not None and past_key_values is not None:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    attention_mask.new_ones((attention_mask.shape[0], 1)),
                ],
                dim=-1,
            )

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(self, input_ids=None, labels=None, **kwargs):
        outputs = self.accustum_forward(input_ids, **kwargs)
        return (
            outputs.last_hidden_state,
            outputs.hidden_states,
        )  # final feat, intermidiate feat

    def accustum_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size()
            )
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                # `use_cache=True` is incompatible with gradient checkpointing.
                # Setting `use_cache=False`...
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        all_hidden_states = () if output_hidden_states else None

        for i, (block, past_key_value) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure past_key_value is on same device as hidden_states
                if past_key_value is not None:
                    past_key_value = tuple(
                        past_state.to(hidden_states.device)
                        for past_state in past_key_value
                    )
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    past_key_value=past_key_value,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                if use_cache:
                    if len(outputs) > 2:
                        all_self_attentions = all_self_attentions + (outputs[2],)
                    if self.config.add_cross_attention and len(outputs) > 3:
                        all_cross_attentions = all_cross_attentions + (outputs[3],)
                else:
                    if len(outputs) > 1:
                        all_self_attentions = all_self_attentions + (outputs[1],)
                    if self.config.add_cross_attention and len(outputs) > 2:
                        all_cross_attentions = all_cross_attentions + (outputs[2],)

            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
