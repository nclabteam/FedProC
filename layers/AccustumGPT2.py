import torch
from transformers import GenerationMixin
from transformers.models.gpt2.modeling_gpt2 import GPT2Model


def gpt2_pca_embeddings(model: "AccustumGPT2Model", n_components: int = 500, device=None) -> torch.Tensor:
    """Return PCA-reduced GPT-2 word-token embeddings as a float32 tensor of shape [n_components, vocab_size]."""
    from sklearn.decomposition import PCA

    wte = model.wte.state_dict()["weight"].cpu().numpy()
    pca = PCA(n_components=n_components)
    wte_pca = pca.fit_transform(wte.T)
    t = torch.tensor(wte_pca, dtype=torch.float32)
    return t.to(device) if device is not None else t


class AccustumGPT2Model(GPT2Model, GenerationMixin):
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            input_ids = input_ids[:, -1:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1:]
        if attention_mask is not None and past_key_values is not None:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(self, input_ids=None, labels=None, **kwargs):
        kwargs.pop("head_mask", None)
        kwargs.pop("output_attentions", None)
        kwargs.pop("output_hidden_states", None)
        # Temporarily disable config flags so @capture_outputs never calls
        # maybe_install_capturing_hooks, which would install permanent closures
        # referencing ContextVar — making the model unpickleable by Ray.
        # We collect hidden states ourselves via removable temporary hooks instead.
        orig_hs = self.config.output_hidden_states
        orig_att = self.config.output_attentions
        self.config.output_hidden_states = False
        self.config.output_attentions = False

        captured: list = []

        def _hook(module, args, output):
            if not captured:
                captured.append(args[0])  # embeddings before first block
            captured.append(output if not isinstance(output, tuple) else output[0])

        hooks = [blk.register_forward_hook(_hook) for blk in self.h]
        try:
            out = super().forward(input_ids=input_ids, **kwargs)
        finally:
            for h in hooks:
                h.remove()
            self.config.output_hidden_states = orig_hs
            self.config.output_attentions = orig_att

        if captured:
            captured[-1] = out.last_hidden_state  # tie last to post-ln state
        return out.last_hidden_state, tuple(captured)
