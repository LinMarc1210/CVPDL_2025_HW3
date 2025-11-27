import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np

DATASET_PATH = os.path.join(os.path.dirname(__file__), 'mnist')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 3e-4
BATCH_SIZE = 128
IMG_SIZE = 32
CHANNELS = 1
EPOCHS = 30
T = 1000
NUM_WORKERS = 2

def scale_to_minus_one_one(t):
    return (t * 2) - 1

class DiffusionUtils:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=32, device="cpu"):
        self.noise_steps = noise_steps
        self.img_size = img_size
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,)).to(self.device)

    def sample(self, model, n):
        print(f"Sampling {n} new images...")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, CHANNELS, self.img_size, self.img_size)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (1, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = CHANNELS
        down_channels = (64, 128, 256)
        up_channels = (256, 128, 64)
        out_dim = CHANNELS
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels)-1)
        ])
        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) for i in range(len(up_channels)-1)
        ])
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        residuals = []
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)
        for up in self.ups:
            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1)
            x = up(x, t)
        return self.output(x)


def main():
    print(f"Using Device: {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        scale_to_minus_one_one,
    ])

    try:
        full_dataset = ImageFolder(root=DATASET_PATH, transform=transform)
        print(f"成功載入資料集，共 {len(full_dataset)} 張圖片。")
    except Exception as e:
        print(f"讀取資料失敗: {e}")
        print("請確認 DATASET_PATH 路徑正確，且該路徑下有'子資料夾'包含圖片。")
        return


    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)

    model = SimpleUNet().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    diffusion = DiffusionUtils(noise_steps=T, img_size=IMG_SIZE, device=DEVICE)

    print("開始訓練 DDPM (無圖形介面版)...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss_accum = 0.0
        for i, (images, _) in enumerate(train_loader):
            images = images.to(DEVICE)
            t = diffusion.sample_timesteps(images.shape[0])
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = loss_fn(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item()
        avg_train_loss = train_loss_accum / len(train_loader)

        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(DEVICE)
                t = diffusion.sample_timesteps(images.shape[0])
                x_t, noise = diffusion.noise_images(images, t)
                predicted_noise = model(x_t, t)
                loss = loss_fn(noise, predicted_noise)
                val_loss_accum += loss.item()
        avg_val_loss = val_loss_accum / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            sampled_images = diffusion.sample(model, n=16)
            save_image(sampled_images, f"epoch_{epoch+1}.png")
            print(f"-> 圖片已儲存至: epoch_{epoch+1}.png")

    torch.save(model.state_dict(), "ddpm_mnist_custom.pth")
    print("模型已儲存為 ddpm_mnist_custom.pth")

if __name__ == "__main__":
    main()