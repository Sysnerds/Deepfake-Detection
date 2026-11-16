import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import glob

# ==========================================
# 1. UNIVERSAL DATASET (MULTIPLE REAL DIRS + GAN + DIFFUSION)
# ==========================================
class UniversalDataset(Dataset):
    def __init__(self, real_dirs, gan_dir, diff_dir, img_size=224, max_images=30000):
        # Load real images from MULTIPLE folders (Kaggle + Wiki)
        real_imgs = []
        for r_dir in real_dirs:
            real_imgs.extend(glob.glob(os.path.join(r_dir, "**", "*.jpg"), recursive=True))
            real_imgs.extend(glob.glob(os.path.join(r_dir, "**", "*.png"), recursive=True))
        
        # Get GAN images (Kaggle)
        gan_imgs = glob.glob(os.path.join(gan_dir, "**", "*.jpg"), recursive=True)
        gan_imgs.extend(glob.glob(os.path.join(gan_dir, "**", "*.png"), recursive=True))
        
        # Get Diffusion images (HuggingFace Text2Img)
        diff_imgs = glob.glob(os.path.join(diff_dir, "**", "*.jpg"), recursive=True)[:5000] 
        
        # Balance dataset (Equal number of each class)
        min_len = min(len(real_imgs), len(gan_imgs), len(diff_imgs), max_images // 3)
        random.seed(42)
        
        self.image_paths = (random.sample(real_imgs, min_len) + 
                            random.sample(gan_imgs, min_len) + 
                            random.sample(diff_imgs, min_len))
                            
        self.labels = [0] * min_len + [1] * min_len + [1] * min_len # 0=Real, 1=Fake
        self.img_size = img_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: 
            img_bgr = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # FFT Computation
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = self.normalize(img_tensor)
        fft_tensor = torch.from_numpy(magnitude).unsqueeze(0).float() / 255.0
        
        combined_tensor = torch.cat((img_tensor, fft_tensor), dim=0)
        return combined_tensor, torch.tensor(label, dtype=torch.float32)

# ==========================================
# 2. DUAL-INPUT RESNET50
# ==========================================
def get_dual_input_resnet():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    original_conv = model.conv1
    new_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = original_conv.weight
        new_conv.weight[:, 3, :, :] = original_conv.weight.mean(dim=1)
        
    model.conv1 = new_conv
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train():
    # Mix of Kaggle and Wiki real faces to prevent domain shift
    REAL_DIRS = [
        r"C:\Users\mda16\Downloads\Dataset\real_vs_fake\real-vs-fake\train\real",
        r"C:\Users\mda16\OneDrive\Desktop\DeepfakeDetector\external_data\wiki\wiki"
    ]
    GAN_DIR = r"C:\Users\mda16\Downloads\Dataset\real_vs_fake\real-vs-fake\train\fake"
    # We train on the HuggingFace Text2Img (Stable Diffusion) data
    DIFF_DIR = r"C:\Users\mda16\OneDrive\Desktop\DeepfakeDetector\external_data\text2img\text2img"

    BATCH_SIZE = 4
    ACCUM_STEPS = 8
    EPOCHS = 2
    LR = 0.0001

    print("Loading Universal Data (Kaggle Real + Wiki Real + GAN + Diffusion)...", flush=True)
    
    dataset = UniversalDataset(REAL_DIRS, GAN_DIR, DIFF_DIR, img_size=224, max_images=30000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    print("Initializing Dual-Input ResNet50...", flush=True)
    model = get_dual_input_resnet().cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    print(f"Training on {len(dataset)} images...", flush=True)
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad()
        
        loop = tqdm(dataloader, leave=True)
        for step, (images, labels) in enumerate(loop):
            images, labels = images.cuda(), labels.cuda().unsqueeze(1)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) / ACCUM_STEPS
                
            scaler.scale(loss).backward()
            
            if (step + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item() * ACCUM_STEPS
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=f"{loss.item()*ACCUM_STEPS:.4f}", acc=f"{correct/total:.3f}")
        
        print(f"Epoch {epoch+1}: Loss {running_loss/len(dataloader):.4f}, Acc {100*correct/total:.2f}%", flush=True)
        
    torch.save(model.state_dict(), "universal_detector.pth")
    print("Training Complete! Saved as universal_detector.pth", flush=True)

if __name__ == "__main__":
    train()