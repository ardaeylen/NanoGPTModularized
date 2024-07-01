import torch
import torch.nn.functional as F
from dataset import TinyShakespeare
from torch.utils.data import DataLoader
from model import BigramLanguageModel
from tqdm import tqdm
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    for samples, targets in tqdm(data_loader):
        samples = samples.to(device)
        targets = targets.to(device)
        logits = model(samples)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
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
    model = BigramLanguageModel(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    device = torch.device("cuda")
    model.to(device)
    for epoch in range(epochs):
        train_one_epoch(model, optimizer, train_loader, device)
        validate(model, val_loader, device)