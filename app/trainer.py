import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from app.models import Generator

def train_gan(epochs=1, latent_dim=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    loader = DataLoader(datasets.MNIST("./data", train=True, download=True, transform=transform),
                        batch_size=64, shuffle=True)
    G, D = Generator(latent_dim).to(device), nn.Sequential(
        nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid()).to(device)
    loss_fn, opt_G, opt_D = nn.BCELoss(), optim.Adam(G.parameters(), 1e-3), optim.Adam(D.parameters(), 1e-3)
    for epoch in range(epochs):
        for x, _ in loader:
            x = x.view(-1, 784).to(device)
            real, fake = torch.ones(x.size(0),1).to(device), torch.zeros(x.size(0),1).to(device)
            z = torch.randn(x.size(0), latent_dim).to(device)
            # Train D
            opt_D.zero_grad()
            loss_D = (loss_fn(D(x), real) + loss_fn(D(G(z).view(-1,784).detach()), fake))/2
            loss_D.backward(); opt_D.step()
            # Train G
            opt_G.zero_grad()
            loss_G = loss_fn(D(G(z).view(-1,784)), real)
            loss_G.backward(); opt_G.step()
        print(f"Epoch {epoch+1} | G_loss {loss_G:.4f} | D_loss {loss_D:.4f}")
    torch.save(G.state_dict(), "artifacts/generator_mnist.pt")
