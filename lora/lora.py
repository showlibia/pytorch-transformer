import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, merge: bool, rank: int = 16, lora_alpha: int = 16,  dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.merge = merge
        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(in_features, out_features)
        if rank > 0:
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.scale = lora_alpha / rank
            self.linear.weight.requires_grad = False

        self.initial_weights()

    def initial_weights(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        if self.rank > 0 and self.merge:
            output = F.linear(x, self.linear.weight + self.scale * (self.lora_B @ self.lora_A), self.linear.bias)   
        else:
            output = self.linear(x)
        return self.dropout(output)
