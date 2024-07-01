import random
import torch
import torch.nn.functional as F
from model import BigramLanguageModel
from dataset import TinyShakespeare

def infer(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        prediction = model(idx) # Predict the next token according to given current context. (B, T, C)
        prediction = prediction[:, -1, :] # Acquire the last token of the sequence (current prediction). (B , C)
        probs = F.softmax(prediction, dim=-1) # Get the index of token in vocabulary according to the softmax probabilities predicted
                                              # by the model
        idx_next = torch.multinomial(probs, num_samples=1) # Sample "num_samples" number of samples from probabilities. (B, 1)
        idx = torch.cat([idx, idx_next], dim = 1) # Concatenate the previous and current predictions along time axis. (B, T + 1)
    return idx

if __name__ == "__main__":
    vocab_size = 65
    model = BigramLanguageModel(vocab_size = vocab_size)
    idx = torch.randint(0, vocab_size, (1, 1))
    max_new_tokens = 85
    charset = TinyShakespeare("input.txt", 64, "val").charset
    itos = {idx: element for idx, element in enumerate(charset)}

    predictions = infer(model, idx, max_new_tokens)[0] # Get the predictions and unplug the batch dimension.

    print(''.join(itos[int(idx)] for idx in predictions))