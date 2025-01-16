import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoAttention(nn.Module):
    """
    Source:https://github.com/Levi-Ackman/Leddam/blob/main/layers/Leddam.py
    """

    def __init__(self, P, d_model, proj_dropout=0.2):
        """
        Initialize the Auto-Attention module.

        Args:
            d_model (int): The input and output dimension for queries, keys, and values.
        """
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_projector = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Dropout(proj_dropout)
        )
        self.P = P
        self.scale = nn.Parameter(torch.tensor(d_model**-0.5), requires_grad=False)

    def auto_attention(self, inp):
        """
        Perform auto-attention mechanism on the input.

        Args:
            inp (torch.Tensor): Input data of shape [B, N, T], where B is the batch size,
                               N is the number of features, and T is the sequence length.
        Returns:
            output (torch.Tensor): Output after auto-attention.
        """
        # Separate query and key
        query = self.W_Q(inp[:, :, 0, :].unsqueeze(-2))  # Query
        keys = self.W_K(inp)  # Keys
        values = self.W_V(inp)  # Values

        # Calculate dot product
        attn_scores = torch.matmul(query, keys.transpose(-2, -1)) * (self.scale)

        # Normalize attention scores
        attn_scores = F.softmax(attn_scores, dim=-1)

        # Weighted sum
        output = torch.matmul(attn_scores, values)

        return output

    def forward(self, inp):
        """
        Forward pass of the Auto-Attention module.

        Args:
            P (int): The period for autoregressive behavior.
            inp (torch.Tensor): Input data of shape [B, T, N], where B is the batch size,
                               T is the sequence length, and N is the number of features.

        Returns:
            output (torch.Tensor): Output after autoregressive self-attention.
        """
        # Permute the input for further processing
        inp = inp.permute(0, 2, 1)  # [B, T, N] -> [B, N, T]

        T = inp.size(-1)

        cat_sequences = [inp]
        index = int(T / self.P) - 1 if T % self.P == 0 else int(T / self.P)

        for i in range(index):
            end = (i + 1) * self.P

            # Concatenate sequences to support autoregressive behavior
            cat_sequence = torch.cat([inp[:, :, end:], inp[:, :, 0:end]], dim=-1)
            cat_sequences.append(cat_sequence)

        # Stack the concatenated sequences
        output = torch.stack(cat_sequences, dim=-1)

        # Permute the output for attention calculation
        output = output.permute(0, 1, 3, 2)

        # Apply autoregressive self-attention
        output = self.auto_attention(output).squeeze(-2)
        output = self.out_projector(output).permute(0, 2, 1)

        return output
