import os, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision, torchvision.transforms as T
from models.cnn import SimpleCNN

def get_loaders(batch_size=128):
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
    ])
    train = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=tfm)
    test  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    return DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=2), \
           DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=2)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum/total, correct/total

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)
    train_loader, test_loader = get_loaders()
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 3
    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = eval_epoch(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | test loss {te_loss:.4f} acc {te_acc:.3f}")

    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/cnn_cifar10.pt")
    print("Saved weights to artifacts/cnn_cifar10.pt")
