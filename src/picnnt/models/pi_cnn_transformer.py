from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_positional_encoding(seq_len: int, d_model: int, device=None) -> torch.Tensor:
    pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        q = self.w_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.w_o(out)
        return out, attn


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.mhsa(x)
        h1 = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(h1)
        h2 = self.norm2(h1 + self.dropout(ffn_out))
        return h2, attn_weights


class PICNNTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        use_cnn_front_end: bool = True,
        use_positional_encoding: bool = True,
        pooling: str = "mean",
        max_seq_len: int = 200,
    ):
        super().__init__()
        assert pooling in ("mean", "max")
        self.use_cnn_front_end = use_cnn_front_end
        self.use_positional_encoding = use_positional_encoding
        self.pooling = pooling
        self.d_model = d_model

        if use_cnn_front_end:
            self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(32)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(64)
            cnn_out_dim = 64
        else:
            cnn_out_dim = input_dim

        self.embed = nn.Linear(cnn_out_dim, d_model)

        self.register_buffer(
            "pos_encoding", sinusoidal_positional_encoding(max_seq_len, d_model), persistent=False
        )

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, ffn_dim, dropout) for _ in range(num_encoder_layers)
        ])

        self.out_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if self.use_cnn_front_end:
            h = x.transpose(1, 2)
            h = torch.relu(self.bn1(self.conv1(h)))
            h = torch.relu(self.bn2(self.conv2(h)))
            h = h.transpose(1, 2)
        else:
            h = x

        e = self.embed(h)
        if self.use_positional_encoding:
            L = e.shape[1]
            e = e + self.pos_encoding[:L].unsqueeze(0)

        attn_weights_per_layer = []
        for layer in self.layers:
            e, attn = layer(e)
            attn_weights_per_layer.append(attn)

        if self.pooling == "mean":
            pooled = e.mean(dim=1)
        else:
            pooled = e.max(dim=1).values

        y_hat = self.out_head(pooled).squeeze(-1)

        if return_attention:
            return y_hat, attn_weights_per_layer
        return y_hat

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
