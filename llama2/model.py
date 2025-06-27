from ast import Mod
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for kvcache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None # type: ignore

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, seq_len, dim) * (B, seq_len, 1) -> (B, seq_len, dim)
        # rsqrt = 1 / sqrt
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (dim) * (B, seq_len, dim) -> (B, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)

def repeat_kv(x: torch.Tensor, n_rep: int):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, seq_len, n_kv_heads, 1, head_dim)
        x[:, :, :, None, :]
        # (B, seq_len, n_kv_heads, n_rep, head_dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, seq_len, n_kv_heads * n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # number of heads for keys and values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # number of heads for queries
        self.n_q_heads= args.n_heads
        # how many times the keys and values should be repeated for GQA
        self.n_rep = self.n_q_heads // self.n_kv_heads
        # numbers of dimensions for each head
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        # (B, 1, dim) -> (B, 1, H_Q * head_dim)
        xq = self.wq(x)
        # (B, 1, dim) -> (B, 1, H_KV * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (B, 1, H_Q * head_dim) -> (B, 1, H_Q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_q_heads, self.head_dim)
        # (B, 1, H_KV * head_dim) -> (B, 1, H_KV, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply rotary to queries and keys
        # (B, 1, H_Q, head_dim) -> (B, 1, H_Q, head_dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device) # type: ignore
        # (B, 1, H_KV, head_dim) -> (B, 1, H_KV, head_dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device) # type: ignore

        # replace the entry in the cache
        self.cache_k[:batch_size, start_pos: start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos: start_pos + seq_len] = xv

        # (B, seq_len_KV, H_KV, head_dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # Apply GQA
        # (B, seq_len_KV, H_KV, head_dim) -> (B, seq_len_KV, H_Q, head_dim)
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (B, seq_len, H_Q, head_dim) -> (B, H_Q, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        # (B, seq_len_KV, H_Q, head_dim) -> (B, H_Q, seq_len_KV, head_dim)
        keys = keys.transpose(1, 2) # type: ignore
        # (B, seq_len_KV, H_Q, head_dim) -> (B, H_Q, seq_len_KV, head_dim)
        values = values.transpose(1, 2) # type: ignore

        # (B, H_Q, seq_len, head_dim) @ (B, H_Q, head_dim, seq_len_KV) -> (B, H_Q, seq_len, seq_len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # (B, H_Q, seq_len, seq_len_KV) -> (B, H_Q, seq_len, seq_len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, seq_len, seq_len_KV) @ (B, H_Q, seq_len_KV, head_dim) -> (B, H_Q, seq_len, head_dim)
        output = torch.matmul(scores, values)
        # (B, H_Q, seq_len, head_dim) -> (B, seq_len, H_Q, head_dim) -> (B, seq_len, H_Q * head_dim = dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1= nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # FFN = (SwiGLU(xW1)*xV)W2
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.dim // self.n_heads

        # Normalization before the attention, after input embedding
        self.attention_norm = RMSNorm(self.dim, eps=args.norm_eps)
        self.attention = SelfAttention(args)
        # Normalization before FFN
        self.ffn_norm = RMSNorm(self.dim, eps=args.norm_eps)
        self.feed_forward = FeedForward(args)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, seq_len, dim) + (B, seq_len, dim) -> (B, seq_len, dim)
        # Normalize the input embedding, then compute the attention with rotary embeddings
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        # Normalize the attention output, then apply the feed-forward network
        output = h + self.feed_forward.forward(self.ffn_norm(h))
        return output

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta = 10000.0):
    assert head_dim % 2 == 0, "Dimension must be even."
    # Build the theta parameters
    # theta_i = 10000 ^ (-2(i-1)/dim) for i = [1, 2, 3..., dim/2]
    # (dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # Construct the positions
    # m is [0, 1, 2, ..., seq_len-1]
    # (seq_len)
    m = torch.arange(seq_len, device=device)
    # (seq_len, dim/2)
    # freqs is the outer product of m and theta
    # freqs[i, j] = m[i] * theta[j]
    freqs = torch.outer(m, theta).float()
    # compute complex numbers in the polar from c = R * exp(i * m * theta), R = 1
    # exp(i * theta) = cos(theta) + i * sin(theta)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # Separate the last dimension into pairs, representing real and imaginary parts
    # Two consecutive values will become a single complex number
    # For instance, the last dimension of x maybe as [x1, x2, x3, x4] -> [[x1+x2*i], [x3+x4*i]]
    # (B, seq_len, H, head_dim) -> (B, seq_len, H, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape the freqs_complex to match the x_complex shape
    # (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # multiply each complex number in x_complex by the corresponding frequency
    # which results in a rotation of the complex number
    # x * freqs_complex = [[x1+x2*i], [x3+x4*i]] * [cos(m1*theta1) + i*sin(m1*theta1), cos(m1*theta2) + i*sin(m1*theta2)]
    # (B, seq_len, H, head_dim / 2) * (1, seq_len, 1, head_dim / 2) = (B, seq_len, H, head_dim / 2)
    x_rotated = x_complex * freqs_complex
    # Convert the complex numbers back to the real
    # (B, seq_len, H, head_dim / 2) -> (B, seq_len, H, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # Reshape back to the original shape
    # (B, seq_len, H, head_dim / 2, 2) -> (B, seq_len, H, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for idx in range(self.n_layers):
            self.layers.append(DecoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2,
            device=self.args.device
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (batch, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed."

        # (batch, seq_len) -> (batch, seq_len, dim)
        h = self.tok_embeddings(tokens)

        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        
        h = self.norm(h)
        output = self.output(h).float()
        return output
