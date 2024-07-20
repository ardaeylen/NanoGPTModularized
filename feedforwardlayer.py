import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, emb_size):
        super(FeedForward, self).__init__()
        self.emb_size = emb_size
        # Just for additional non-linearity for the model to learn deep features. But notice that feedforward here
        # per token level (B, T, Emb_size) -> (B, T, Emb_size) which is why we don't flatten the input features.
        self.mlp = nn.Sequential(nn.Linear(in_features=self.emb_size, out_features=self.emb_size), nn.ReLU())

    def forward(self, x):
        return self.mlp(x)
