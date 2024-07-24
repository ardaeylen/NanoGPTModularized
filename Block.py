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
        # This layer norm actually is a per token normalization which normalizes every token
        # along it's embedding dimension instead of batch dimension in Batch Normalization. Of
        # course layer norm has gamma and beta trainable parameters the layer norm will eventually
        # create outputs that might not be gaussian but the optimization will determine that.
        self.layer_norm1 = nn.LayerNorm(self.embedding_size)
        self.layer_norm2 = nn.LayerNorm(self.embedding_size)

    def forward(self, x):
        x = x + self.multihead_self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x

if __name__ == "__main__":
    block = Block(256, 16, 64)
    device = torch.device("cuda")
    block.to(device)
    rand_input = torch.randn(8, 64, 256).to(device)

    out = block(rand_input)
    print(out.shape)