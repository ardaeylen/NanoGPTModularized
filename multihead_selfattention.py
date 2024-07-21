import torch
import torch.nn as nn
from self_attention import SelfAttentionKarpathy


# Multi-head attention is just computing multiple attentions in parallel and concatenating information.
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, context_length):
        super(MultiHeadAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.context_length = context_length
        self.head_size = self.embedding_size // self.num_heads
        self.heads = nn.ModuleList([SelfAttentionKarpathy(embedding_size=embedding_size, head_size=self.head_size,
                                                          context_length=self.context_length) for _ in range(self.num_heads)])
        self.proj = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        # Simply calculate multiple attentions in parallel and concatenate the result in embedding dimension.
        x = torch.cat([head(x) for head in self.heads], dim = -1)
        x = self.proj(x)
        return x