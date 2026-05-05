import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()
        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(
            temperature=d_k**0.5, attn_dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            if len(mask.size()) == 3:
                mask = mask.unsqueeze(1)

        output, attn = self.attention(q, k, v, mask=mask)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()
        self.normalize_before = normalize_before
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)
        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual
        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


# ---------------------------------------------------------------------------
# Encoder / Decoder layers
# ---------------------------------------------------------------------------


class PyraformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True
    ):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout,
            normalize_before=normalize_before,
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before
        )

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class PyraformerDecoderLayer(nn.Module):
    def __init__(
        self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True
    ):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout,
            normalize_before=normalize_before,
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before
        )

    def forward(self, Q, K, V, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(Q, K, V, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


# ---------------------------------------------------------------------------
# CSCM (Cross-Scale Construction Module)
# ---------------------------------------------------------------------------


class ConvLayer(nn.Module):
    def __init__(self, c_in, window_size):
        super().__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=window_size,
            stride=window_size,
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Conv_Construct(nn.Module):
    """Convolution CSCM"""

    def __init__(self, d_model, window_size, d_inner):
        super().__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList(
                [ConvLayer(d_model, window_size) for _ in range(3)]
            )
        else:
            self.conv_layers = nn.ModuleList(
                [ConvLayer(d_model, ws) for ws in window_size[:3]]
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.permute(0, 2, 1)
        all_inputs.append(enc_input)
        for conv in self.conv_layers:
            enc_input = conv(enc_input)
            all_inputs.append(enc_input)
        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)
        return all_inputs


class Bottleneck_Construct(nn.Module):
    """Bottleneck convolution CSCM"""

    def __init__(self, d_model, window_size, d_inner):
        super().__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList(
                [ConvLayer(d_inner, window_size) for _ in range(3)]
            )
        else:
            self.conv_layers = nn.ModuleList(
                [ConvLayer(d_inner, ws) for ws in window_size]
            )
        self.up = nn.Linear(d_inner, d_model)
        self.down = nn.Linear(d_model, d_inner)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        temp_input = self.down(enc_input).permute(0, 2, 1)
        all_inputs = []
        for conv in self.conv_layers:
            temp_input = conv(temp_input)
            all_inputs.append(temp_input)
        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.up(all_inputs)
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)
        all_inputs = self.norm(all_inputs)
        return all_inputs


class MaxPooling_Construct(nn.Module):
    """Max pooling CSCM"""

    def __init__(self, d_model, window_size, d_inner):
        super().__init__()
        if not isinstance(window_size, list):
            self.pooling_layers = nn.ModuleList(
                [nn.MaxPool1d(kernel_size=window_size) for _ in range(3)]
            )
        else:
            self.pooling_layers = nn.ModuleList(
                [nn.MaxPool1d(kernel_size=ws) for ws in window_size[:3]]
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.transpose(1, 2).contiguous()
        all_inputs.append(enc_input)
        for layer in self.pooling_layers:
            enc_input = layer(enc_input)
            all_inputs.append(enc_input)
        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)
        return all_inputs


class AvgPooling_Construct(nn.Module):
    """Average pooling CSCM"""

    def __init__(self, d_model, window_size, d_inner):
        super().__init__()
        if not isinstance(window_size, list):
            self.pooling_layers = nn.ModuleList(
                [nn.AvgPool1d(kernel_size=window_size) for _ in range(3)]
            )
        else:
            self.pooling_layers = nn.ModuleList(
                [nn.AvgPool1d(kernel_size=ws) for ws in window_size[:3]]
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.transpose(1, 2).contiguous()
        all_inputs.append(enc_input)
        for layer in self.pooling_layers:
            enc_input = layer(enc_input)
            all_inputs.append(enc_input)
        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)
        return all_inputs


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------


def get_mask(input_size, window_size, inner_size, device):
    """Get the attention mask of PAM-Naive"""
    all_size = [input_size]
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])
        all_size.append(layer_size)

    seq_length = sum(all_size)
    mask = torch.zeros(seq_length, seq_length, device=device)

    # intra-scale mask
    inner_window = inner_size // 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = max(i - inner_window, start)
            right_side = min(i + inner_window + 1, start + all_size[layer_idx])
            mask[i, left_side:right_side] = 1

    # inter-scale mask
    for layer_idx in range(1, len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = (start - all_size[layer_idx - 1]) + (i - start) * window_size[
                layer_idx - 1
            ]
            if i == (start + all_size[layer_idx] - 1):
                right_side = start
            else:
                right_side = (start - all_size[layer_idx - 1]) + (
                    i - start + 1
                ) * window_size[layer_idx - 1]
            mask[i, left_side:right_side] = 1
            mask[left_side:right_side, i] = 1

    mask = (1 - mask).bool()
    return mask, all_size


def refer_points(all_sizes, window_size, device):
    """Gather features from PAM's pyramid sequences"""
    input_size = all_sizes[0]
    indexes = torch.zeros(input_size, len(all_sizes), device=device)

    for i in range(input_size):
        indexes[i][0] = i
        former_index = i
        for j in range(1, len(all_sizes)):
            start = sum(all_sizes[:j])
            inner_layer_idx = former_index - (start - all_sizes[j - 1])
            former_index = start + min(
                inner_layer_idx // window_size[j - 1], all_sizes[j] - 1
            )
            indexes[i][j] = former_index

    indexes = indexes.unsqueeze(0).unsqueeze(3)
    return indexes.long()


def get_subsequent_mask(input_size, window_size, predict_step, truncate):
    """Get causal attention mask for decoder."""
    if truncate:
        mask = torch.zeros(predict_step, input_size + predict_step)
        for i in range(predict_step):
            mask[i][: input_size + i + 1] = 1
        mask = (1 - mask).bool().unsqueeze(0)
    else:
        all_size = [input_size]
        for i in range(len(window_size)):
            layer_size = math.floor(all_size[i] / window_size[i])
            all_size.append(layer_size)
        all_size = sum(all_size)
        mask = torch.zeros(predict_step, all_size + predict_step)
        for i in range(predict_step):
            mask[i][: all_size + i + 1] = 1
        mask = (1 - mask).bool().unsqueeze(0)
    return mask


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class Predictor(nn.Module):
    def __init__(self, dim, num_types):
        super().__init__()
        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data):
        return self.linear(data)


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------


class PyraformerEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.d_model = opt.d_model
        self.window_size = opt.window_size
        self.truncate = opt.truncate

        self.mask, self.all_size = get_mask(
            opt.input_len, opt.window_size, opt.inner_size, opt.device
        )

        self.decoder_type = opt.decoder
        if opt.decoder == "FC":
            self.indexes = refer_points(self.all_size, opt.window_size, opt.device)

        d_k = d_v = opt.d_model // opt.n_head
        self.layers = nn.ModuleList(
            [
                PyraformerEncoderLayer(
                    opt.d_model,
                    opt.d_inner_hid,
                    opt.n_head,
                    d_k,
                    d_v,
                    dropout=opt.dropout,
                    normalize_before=False,
                )
                for _ in range(opt.n_layer)
            ]
        )

        cscm_map = {
            "Conv_Construct": Conv_Construct,
            "Bottleneck_Construct": Bottleneck_Construct,
            "MaxPooling_Construct": MaxPooling_Construct,
            "AvgPooling_Construct": AvgPooling_Construct,
        }
        self.conv_layers = cscm_map[opt.CSCM](
            opt.d_model, opt.window_size, opt.d_bottleneck
        )

    def forward(self, x_enc):
        seq_enc = x_enc
        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        seq_enc = self.conv_layers(seq_enc)

        for layer in self.layers:
            seq_enc, _ = layer(seq_enc, mask)

        if self.decoder_type == "FC":
            indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(
                seq_enc.device
            )
            indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
            all_enc = torch.gather(seq_enc, 1, indexes)
            seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)
        elif self.decoder_type == "attention" and self.truncate:
            seq_enc = seq_enc[:, : self.all_size[0]]

        return seq_enc


class PyraformerDecoder(nn.Module):
    def __init__(self, opt, mask):
        super().__init__()
        self.mask = mask
        d_k = d_v = opt.d_model // opt.n_head

        self.layers = nn.ModuleList(
            [
                PyraformerDecoderLayer(
                    opt.d_model,
                    opt.d_inner_hid,
                    opt.n_head,
                    d_k,
                    d_v,
                    dropout=opt.dropout,
                    normalize_before=False,
                )
                for _ in range(2)
            ]
        )

    def forward(self, dec_input, refer):
        dec_enc = dec_input
        dec_enc, _ = self.layers[0](dec_enc, refer, refer)
        refer_enc = torch.cat([refer, dec_enc], dim=1)
        mask = self.mask.repeat(len(dec_enc), 1, 1).to(dec_enc.device)
        dec_enc, _ = self.layers[1](dec_enc, refer_enc, refer_enc, slf_attn_mask=mask)
        return dec_enc
