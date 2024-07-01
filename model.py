import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super(BigramLanguageModel, self).__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.embedding_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, self.embedding_size)

    def forward(self, idx):
        # idx and targets are both (B, T) tensor of integers.
        # Now every single integer (token) in idx is mapped to corresponding embedding according to index of
        # the given sample token in the vocabulary.
        logits = self.token_embedding_table(idx) # (Batch , Time (Sequence Length), C (Embedding Size))
        return logits

if __name__ == "__main__":
    bigram_model = BigramLanguageModel(65)
    out = bigram_model(torch.randint(0, 65, (8,128)))
    print(out.shape) # Now we got vocab size length embeddings for every (8,128) positions in the input sample.