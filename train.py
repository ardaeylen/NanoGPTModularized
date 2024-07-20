import torch
import torch.nn.functional as F
from dataset import TinyShakespeare
from torch.utils.data import DataLoader
from model import BigramLanguageModel
from tqdm import tqdm
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
    running_accuracy = 0
    model.eval()
    with torch.no_grad():
        for sample, target in tqdm(data_loader):
            sample = sample.to(device)
            target = target.to(device)
            output = model(sample)
            output = torch.argmax(output, -1)
            running_accuracy += torch.sum(output == target) / (target.shape[1] * target.shape[0])
    print(running_accuracy / len(data_loader))

if __name__ == "__main__":
    train_dataset = TinyShakespeare("input.txt", context_len=64, split="train")
    val_dataset = TinyShakespeare("input.txt", context_len=64, split="val")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    epochs = 100
    vocab_size = len(train_dataset.charset)
    embedding_size = 256
    num_heads = 16
    context_length = train_dataset.context_length
    model = BigramLanguageModel(vocab_size, emb_size=embedding_size, block_size=context_length, num_heads=num_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    device = torch.device("cuda")
    model.to(device)
    for epoch in range(epochs):
        epoch_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(epoch_loss)
        validate(model, val_loader, device)