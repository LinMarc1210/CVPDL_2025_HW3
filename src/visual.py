import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

try:
    from train import SimpleUNet, DiffusionUtils, IMG_SIZE, DEVICE, T, CHANNELS
except ImportError:
    raise ImportError("找不到 train.py，請確保 visual.py 與 train.py 位於同一目錄下。")

MODEL_PATH = "ddpm_mnist_custom.pth" 
OUTPUT_FILENAME = "diffusion_process_8x8.png"
NUM_SAMPLES = 8
NUM_SNAPSHOTS = 8 

def sample_and_capture_snapshots(model, diffusion, n=8):
    model.eval()
    print(f"Sampling {n} images and capturing intermediate steps...")
    
    capture_steps = torch.linspace(diffusion.noise_steps - 1, 0, NUM_SNAPSHOTS).long().to(DEVICE)
    capture_steps_set = set(capture_steps.tolist())
    
    x = torch.randn((n, CHANNELS, diffusion.img_size, diffusion.img_size)).to(DEVICE)
    
    snapshots = []
    
    if (diffusion.noise_steps - 1) in capture_steps_set:
        snapshots.append(x.cpu())

    with torch.no_grad():
        for i in reversed(range(1, diffusion.noise_steps)):
            t = (torch.ones(n) * i).long().to(DEVICE)
            predicted_noise = model(x, t)
            
            alpha = diffusion.alpha[t][:, None, None, None]
            alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
            beta = diffusion.beta[t][:, None, None, None]
            
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            
            if (i - 1) in capture_steps_set:
                snapshots.append(x.cpu())

    return snapshots

def create_grid_image(snapshots):
    stacked = torch.stack(snapshots)
    # permuted = stacked.permute(1, 0, 2, 3, 4)
    grid_tensor = stacked.reshape(-1, CHANNELS, IMG_SIZE, IMG_SIZE)
    
    if IMG_SIZE != 28:
        grid_tensor = transforms.Resize((28, 28))(grid_tensor)
        
    if grid_tensor.shape[1] == 1:
        grid_tensor = grid_tensor.repeat(1, 3, 1, 1)

    grid_tensor = (grid_tensor.clamp(-1, 1) + 1) / 2
    
    save_image(grid_tensor, OUTPUT_FILENAME, nrow=NUM_SNAPSHOTS, padding=2)
    print(f"擴散過程圖已儲存至: {OUTPUT_FILENAME}")

def main():
    print(f"Using Device: {DEVICE}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"找不到模型檔案: {MODEL_PATH}")
        return

    model = SimpleUNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    diffusion = DiffusionUtils(noise_steps=T, img_size=IMG_SIZE, device=DEVICE)

    snapshots = sample_and_capture_snapshots(model, diffusion, n=NUM_SAMPLES)
    
    create_grid_image(snapshots)

if __name__ == "__main__":
    main()