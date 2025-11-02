import torch, torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from models.gan import Generator, Discriminator
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
epochs, batch_size, noise_dim, lr = 5, 128, 100, 2e-4

tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=tfm, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

G, D = Generator(noise_dim).to(device), Discriminator().to(device)
criterion = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

os.makedirs("artifacts", exist_ok=True)

for epoch in range(epochs):
    for real, _ in loader:
        real = real.to(device)
        bs = real.size(0)

        
        z = torch.randn(bs, noise_dim, device=device)
        fake = G(z)
        D_real, D_fake = D(real).view(-1), D(fake.detach()).view(-1)
        loss_D = (criterion(D_real, torch.ones_like(D_real)) + criterion(D_fake, torch.zeros_like(D_fake))) / 2
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        
        D_fake = D(fake).view(-1)
        loss_G = criterion(D_fake, torch.ones_like(D_fake))
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()

    print(f"Epoch {epoch+1}/{epochs} | Loss_D={loss_D:.3f}, Loss_G={loss_G:.3f}")
    save_image(fake[:25], f"artifacts/fake_{epoch+1}.png", nrow=5, normalize=True)

torch.save(G.state_dict(), "artifacts/generator_mnist.pt")
print("Training done!")
