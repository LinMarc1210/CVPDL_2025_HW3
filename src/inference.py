import os
import torch
import math
from tqdm import tqdm
from torchvision.utils import save_image
import torchvision.transforms as transforms

try:
    from train import SimpleUNet, DiffusionUtils, IMG_SIZE, DEVICE, T, CHANNELS
except ImportError:
    raise ImportError("找不到 train.py，請確保 inference.py 與 train.py 位於同一目錄下。")

MODEL_PATH = "ddpm_mnist_custom.pth"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'generated_images_10k')
BATCH_SIZE = 64
TOTAL_IMAGES = 10000

def load_trained_model(model_path, device):
    print(f"Loading model from {model_path}...")
    model = SimpleUNet().to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型檔案: {model_path}，請確認是否已完成訓練。")
        
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model

def generate_and_save_images(model, diffusion, num_images, batch_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Cleaning output directory: {output_dir}")
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))

    num_batches = math.ceil(num_images / batch_size)
    print(f"準備生成 {num_images} 張圖片，共需 {num_batches} 個 Batch。")

    generated_count = 0
    resize_transform = transforms.Resize((28, 28))
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Generating"):
            current_batch_size = min(batch_size, num_images - generated_count)
            if current_batch_size <= 0:
                break

            images = diffusion.sample(model, n=current_batch_size)

            if images.shape[-1] != 28:
                images = resize_transform(images)

            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

            for j in range(len(images)):
                filename = f"{generated_count + 1:05d}.png"
                save_path = os.path.join(output_dir, filename)
                save_image(images[j], save_path)
                generated_count += 1

def main():
    print(f"Using Device: {DEVICE}")
    
    diffusion = DiffusionUtils(noise_steps=T, img_size=IMG_SIZE, device=DEVICE)
    model = load_trained_model(MODEL_PATH, DEVICE)

    generate_and_save_images(
        model=model,
        diffusion=diffusion,
        num_images=TOTAL_IMAGES,
        batch_size=BATCH_SIZE,
        output_dir=OUTPUT_DIR
    )
    
    print(f"生成完成！圖片已儲存於 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()