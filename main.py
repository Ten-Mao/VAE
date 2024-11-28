import torch
from matplotlib import pyplot as plt

from torch import nn
from torchvision import datasets
from torchvision.transforms import transforms

from VAE import VAE

def train_func(model, train_loader, val_loader, optimizer, epochs):

    min_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, _ in train_loader:
            x = x.to(device).view(-1, 784)
            optimizer.zero_grad()
            x_recon, mu, log_var = model(x)
            mse_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = mse_loss + kld_loss
            loss.backward()
            train_loss += loss.detach().cpu().item()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset)}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device).view(-1, 784)
                x_recon, mu, log_var = model(x)
                mse_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = mse_loss + kld_loss
                val_loss += loss.detach().cpu().item()
        print(f"Val Loss: {val_loss / len(val_loader.dataset)}")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), 'model_pth/vae.pth')


def display_generated_images(model, device, num_samples=10):
    model.eval()

    # 从标准正态分布中采样潜在空间 z
    z = torch.randn(num_samples, model.latent_dim).to(device)  # 生成潜在变量 z

    # 使用解码器生成图像
    with torch.no_grad():
        generated_images = model.decode(z)

    # 将生成的图像从 [-1, 1] 范围转换到 [0, 1] 范围
    generated_images = generated_images.view(-1, 28, 28).cpu().numpy()

    # 显示生成的图像
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 2))
    for i in range(num_samples):
        axes[i].imshow(generated_images[i], cmap='gray')
        axes[i].axis('off')
    plt.show()


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VAE(input_dim=784, hidden_dim=256, latent_dim=2).to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # 标准化
    ])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
    train_func(model, train_loader, val_loader, torch.optim.Adam(model.parameters(), lr=1e-3), epochs=100)
    model = VAE(input_dim=784, hidden_dim=256, latent_dim=2).to(device)
    model.load_state_dict(torch.load('model_pth/vae.pth'))
    display_generated_images(model, device)

