import torch
import torch.nn as nn
from Block import Block
from Block import TransformerBlock


class LanguageModel(nn.Module):

    def __init__(self, vocab_size, n_layer, emb_size=256, query_dim=256, value_dim=256, block_size=64, num_heads=16,
                 dropout_rate=0.2):
        super(LanguageModel, self).__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.embedding_size = emb_size
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.query_dim = query_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.layer_num = n_layer
        self.dropout = dropout_rate
        self.token_embedding_table = nn.Embedding(vocab_size, self.embedding_size)
        self.position_embedding_table = nn.Embedding(self.block_size, self.embedding_size)
        self.blocks = nn.Sequential(*[
            TransformerBlock(context_length=self.block_size, embedding_size=self.embedding_size,
                             query_dim=self.query_dim, value_dim=self.value_dim, num_heads=self.num_heads, dropout_rate=self.dropout) for _ in
            range(self.layer_num)])
        self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_size)  # Final layer norm.
        self.lm_head = nn.Linear(self.embedding_size, self.vocab_size)

    def forward(self, idx):
        # idx is (B, T) tensor of integers.
        # Now every single integer (token) in idx is mapped to corresponding embedding according to index of
        # the given sample token in the vocabulary.
        token_emb = self.token_embedding_table(idx)
        # (Batch , Time (Sequence Length), C (Embedding Size))
        pos_emb = self.position_embedding_table(torch.arange(0, self.block_size, device=torch.device("cuda")))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)  # outputs (Batch, Time (Sequence Length), Vocab Size)

        return logits


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_layer, emb_size=256, block_size=64, num_heads = 16, dropout_rate = 0.2):
        super(BigramLanguageModel, self).__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.embedding_size = emb_size
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_heads = num_heads
        self.layer_num = n_layer
        self.dropout = dropout_rate
        self.token_embedding_table = nn.Embedding(vocab_size, self.embedding_size)
        self.position_embedding_table = nn.Embedding(self.block_size, self.embedding_size)
        self.blocks = nn.Sequential(
            *[Block(self.embedding_size, self.num_heads, self.block_size, self.dropout) for _ in range(self.layer_num)])
        self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_size)  # Final layer norm.
        self.lm_head = nn.Linear(self.embedding_size, self.vocab_size)

    def forward(self, idx):
        # idx is (B, T) tensor of integers.
        # Now every single integer (token) in idx is mapped to corresponding embedding according to index of
        # the given sample token in the vocabulary.
        token_emb = self.token_embedding_table(idx)
        # (Batch , Time (Sequence Length), C (Embedding Size))
        pos_emb = self.position_embedding_table(torch.arange(0, self.block_size, device=torch.device("cuda")))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)  # outputs (Batch, Time (Sequence Length), Vocab Size)

        return logits


if __name__ == "__main__":
    bigram_model = LanguageModel(vocab_size=65, n_layer=3)
    device = torch.device("cuda")
    bigram_model.to(device)
    out = bigram_model(torch.randint(0, 65, (8, 64), device=torch.device("cuda")))
    print(out.shape)  # Now we got vocab size length embeddings for every (8,128) positions in the input sample.
