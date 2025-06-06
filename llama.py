import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.scale = math.sqrt(emb_size)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        # tokens: (batch, seq)
        # output: (batch, seq, emb_size) scaled by sqrt(emb_size)
        return self.embedding(tokens) * self.scale


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        # x: (batch, seq)
        tok_emb = self.token(x)  # (batch, seq, d_model)
        return self.dropout(tok_emb)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # (.., 2*k) -> (.., k), (.., k)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_rotary_pos_emb(dim: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
    cos = emb.cos()[None, None, :, :]  # (1, 1, seq_len, dim)
    sin = emb.sin()[None, None, :, :]  # (1, 1, seq_len, dim)
    return cos, sin


def apply_rotary_pos_emb_single(q, k, cos, sin, seq_len, dtype):
    cos = cos[:, :, :seq_len, :].to(dtype=dtype)
    sin = sin[:, :, :seq_len, :].to(dtype=dtype)
    q_ = (q * cos) + (rotate_half(q) * sin)
    k_ = (k * cos) + (rotate_half(k) * sin)
    return q_, k_


class Attention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.float()
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        p_attn = self.dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, max_seq_len=2048):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(dropout)

        # 함수 기반 RoPE 임베딩을 버퍼에 등록
        cos, sin = build_rotary_pos_emb(self.d_k, max_seq_len)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()
        dtype = query.dtype

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # RoPE 적용
        query, key = apply_rotary_pos_emb_single(query, key, self.cos, self.sin, seq_len, dtype)

        x, attn = self.attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = RMSNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: callable) -> torch.Tensor:
        # Pre-norm → sublayer → dropout → residual add
        return x + self.dropout(sublayer(self.norm(x)))


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.silu(x2)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff * 2, bias=False)
        self.activation = SwiGLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)  # (batch, seq, d_ff*2)
        x = self.activation(x)  # (batch, seq, d_ff)
        x = self.linear2(x)  # (batch, seq, d_model)
        return self.dropout(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadedAttention(heads, d_model, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayers = nn.ModuleList([
            SublayerConnection(d_model, dropout),
            SublayerConnection(d_model, dropout)
        ])

    def forward(self, x, mask=None):
        x = self.sublayers[0](x, lambda _x: self.attn(_x, _x, _x, mask))
        x = self.sublayers[1](x, self.ff)
        return x


class LLaMA(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, heads, hidden_dim, max_len=2048, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.embed = InputEmbedding(vocab_size, dim, dropout)
        self.layers = nn.ModuleList(
            [DecoderBlock(dim, heads, hidden_dim) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.token.embedding.weight

    def forward(self, x):
        batch_size, seq_len = x.size()
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max length {self.max_len}")

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)

        x = self.embed(x)

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)
        logits = self.head(x)
        return logits
