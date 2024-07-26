import torch
import torch.nn.functional as F
import os
from dataset import TinyShakespeare
from torch.utils.data import DataLoader
from model import BigramLanguageModel
from tqdm import tqdm
import argparse
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    running_loss = 0
    for samples, targets in tqdm(data_loader):
        samples = samples.to(device)
        targets = targets.to(device)
        logits = model(samples)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        running_loss += loss.item()
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
    return running_loss/len(data_loader)
def validate(model, data_loader, device):
    valid_loss = 0.0
    model.eval()
    with torch.no_grad():
        for sample, target in tqdm(data_loader):
            sample = sample.to(device)
            target = target.to(device)
            output = model(sample)
            B, T, C = output.shape
            output = output.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(output, target)
            valid_loss += loss.item()
    return valid_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-len", help="Attention context length of the transformer blocks.", type=int, default=128)
    parser.add_argument("--emb-size", help="Token embedding size.", type=int, default = 256)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=0.0003)
    parser.add_argument("--batch", help="Mini batch size", type=int, default = 64)
    parser.add_argument("--num-layers", help="Number of consequtive transformer layers.", type=int, default = 6)
    parser.add_argument("--num-heads", help="Multi-head self attention head number.", type=int, default=16)
    parser.add_argument("--dropout", help="Dropout probability for regularization.", type=float, default= 0.2)
    parser.add_argument("--epochs", help="Number of epochs for training.", type=int, default = 100)
    parser.add_argument("--exp-dir", help="Directory for the best model to be saved.", type=str, default = None)

    args = parser.parse_args()

    # Hyperparameters
    epochs = args.epochs
    vocab_size = 65
    embedding_size = args.emb_size
    num_heads = args.num_heads
    context_length = args.context_len
    learning_rate = args.lr
    batch_size = args.batch
    dropout_prob = args.dropout
    num_transformer_layers = args.num_layers
    exp_dir = args.exp_dir

    train_dataset = TinyShakespeare("input.txt", context_len=context_length, split="train")
    val_dataset = TinyShakespeare("input.txt", context_len=context_length, split="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    if exp_dir is not None:
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

    model = BigramLanguageModel(vocab_size, n_layer=num_transformer_layers, emb_size=embedding_size, block_size=context_length, num_heads=num_heads, dropout_rate = dropout_prob)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda")
    model.to(device)
    min_validation_loss = float("inf")
    for epoch in range(epochs):
        epoch_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Training Loss: {epoch_loss}")
        valid_loss = validate(model, val_loader, device)
        print(f"Validation Loss: {valid_loss}")
        if valid_loss < min_validation_loss:
            min_validation_loss = valid_loss
            if exp_dir is not None:
                torch.save(model.state_dict(), os.path.join(exp_dir, "best.pth"))
