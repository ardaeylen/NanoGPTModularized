import torch.nn as nn
from multihead_selfattention import MultiHeadAttention
from feedforwardlayer import FeedForward
import torch

class Block(nn.Module):
    def __init__(self, embedding_size, num_heads, context_len):
        super(Block, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.context_len = context_len
        self.multihead_self_attention = MultiHeadAttention(embedding_size = self.embedding_size, num_heads = self.num_heads, context_length = self.context_len)
        self.feed_forward = FeedForward(emb_size = self.embedding_size)

    def forward(self, x):
        x = x + self.multihead_self_attention(x)
        x = x + self.feed_forward(x)
        return x

if __name__ == "__main__":
    block = Block(256, 16, 64)
    device = torch.device("cuda")
    block.to(device)
    rand_input = torch.randn(8, 64, 256).to(device)

    out = block(rand_input)
    print(out.shape)