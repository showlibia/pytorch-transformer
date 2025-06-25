from dataclasses import dataclass
from tracemalloc import start
from typing import Optional
import torch
import torch.nn as nn

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for kvcache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

class EncoderBlock(nn.Module):
    pass

class RMSNorm(nn.Module):
    pass

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
        self.n_layers = args.n_layer
        self.token_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for idx in range(self.n_layers):
            self.layers.append(EncoderBlock(args))

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
        h = self.token_embeddings(tokens)

        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        
        h = self.norm(h)
        output = self.output(h).float()
        return output
        