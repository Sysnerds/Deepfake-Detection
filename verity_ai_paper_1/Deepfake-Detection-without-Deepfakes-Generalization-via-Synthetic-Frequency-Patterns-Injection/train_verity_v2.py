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
# 1. FREQUENCY INJECTOR (WITH FOURIER PATTERNS)
# ==========================================
class FrequencyInjector:
    def __init__(self):
        self.patterns = [
            self.pattern_grid, self.pattern_checkerdboard, self.pattern_centered_aura,
            self.pattern_random_points, self.pattern_ring_aura, self.pattern_stripes,
            self.pattern_circles, self.pattern_squares,
            # CRITICAL: Fourier patterns (mimic real generator fingerprints)
            self.pattern_symmetric_points_fourier, self.pattern_symmetric_blobs_fourier,
            self.pattern_symmetric_grid,
        ]

    def apply(self, img):
        pattern_function = random.choice(self.patterns)
        res = pattern_function(cols=img.shape[1], rows=img.shape[0])
        pattern = res[0]
        
        f_pattern = np.fft.fft2(pattern, s=(img.shape[0], img.shape[1]))
        result_channels = []
        
        for channel in range(img.shape[2]):
            f_transform_channel = np.fft.fft2(img[..., channel])
            magnitude_original = np.abs(f_transform_channel)
            phase_original = np.angle(f_transform_channel)
            
            magnitude_pattern = np.abs(f_pattern)
            magnitude_result = magnitude_original + magnitude_pattern 
            
            f_result_channel = magnitude_result * np.exp(1j * phase_original)
            result_channel = np.fft.ifft2(f_result_channel).real
            result_channels.append(result_channel)
            
        result = np.stack(result_channels, axis=-1)
        return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def pattern_grid(self, cols, rows):
        p = np.zeros((rows, cols)); cs = random.randint(10, 30)
        for y in range(0, rows, cs):
            for x in range(0, cols, cs):
                p[y:y+cs, x:x+cs] = random.randint(50, 200)
        return p, True

    def pattern_checkerdboard(self, cols, rows):
        p = np.zeros((rows, cols)); cs = random.randint(10, 30); i = random.randint(150, 255)
        cx, cy = cols // 2, rows // 2; mo = min(cx, cy) - cs
        for y in range(cy - mo, cy + mo, cs*2):
            for x in range(cx - mo, cx + mo, cs*2):
                p[y:y+cs, x:x+cs] = i
        return p, True

    def pattern_centered_aura(self, cols, rows):
        p = np.zeros((rows, cols)); cx, cy = cols//2, rows//2; mr = min(cx, cy) or 1
        y, x = np.ogrid[:rows, :cols]
        d = np.sqrt((x - cx)**2 + (y - cy)**2) / mr
        return np.exp(-5 * d**2), True

    def pattern_random_points(self, cols, rows):
        p = np.zeros((rows, cols))
        for _ in range(random.randint(2, 30)):
            p[random.randint(0, rows-1), random.randint(0, cols-1)] = random.randint(10, 255)
        return p, True

    def pattern_ring_aura(self, cols, rows):
        p = np.zeros((rows, cols)); cx, cy = cols//2, rows//2
        inn = random.randint(20, min(cx, cy) - 20); out = random.randint(inn + 10, min(cx, cy) - 10)
        y, x = np.ogrid[:rows, :cols]
        d = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask = (d >= inn) & (d <= out)
        p[mask] = np.exp(-5 * ((d[mask] - inn) / (out - inn))**2)
        return p, True

    def pattern_stripes(self, cols, rows):
        p = np.zeros((rows, cols)); sw = random.randint(10, 30)
        for _ in range(random.randint(5, 20)):
            sy = random.randint(0, rows - 1); p[sy:sy+sw, :] = random.randint(100, 255)
        return p, True

    def pattern_circles(self, cols, rows):
        p = np.zeros((rows, cols))
        for _ in range(random.randint(2, 20)):
            r = random.randint(10, min(cols, rows) // 4)
            cx, cy = random.randint(r, cols-r), random.randint(r, rows-r)
            y, x = np.ogrid[:rows, :cols]
            p[(x-cx)**2 + (y-cy)**2 <= r**2] = random.randint(50, 200)
        return p, True

    def pattern_squares(self, cols, rows):
        p = np.zeros((rows, cols))
        for _ in range(random.randint(5, 20)):
            s = random.randint(10, 30); cx, cy = random.randint(0, cols-s), random.randint(0, rows-s)
            p[cy:cy+s, cx:cx+s] = random.randint(150, 255)
        return p, True

    def pattern_symmetric_points_fourier(self, cols, rows):
        p = np.zeros((rows, cols))
        for _ in range(random.randint(5, 20)):
            intensity = random.uniform(100, 200)
            px = random.randint(0, cols // 2 - 1); py = random.randint(0, rows // 2 - 1)
            p[py, px] = intensity; p[rows-1-py, cols-1-px] = intensity
            p[py, cols-1-px] = intensity; p[rows-1-py, px] = intensity
        return np.fft.ifft2(p**2).real, True

    def pattern_symmetric_blobs_fourier(self, cols, rows):
        p = np.zeros((rows, cols))
        for _ in range(random.randint(2, 12)):
            intensity = random.uniform(100, 255); size = random.randint(5, 30)
            bx = random.randint(0, cols // 2 - size); by = random.randint(0, rows // 2 - size)
            blob = self._gaussian_blob(size, intensity)
            p[by:by+size, bx:bx+size] = blob
            p[rows-1-by-size:rows-1-by, cols-1-bx-size:cols-1-bx] = blob[::-1, ::-1]
            p[by:by+size, cols-1-bx-size:cols-1-bx] = blob[:, ::-1]
            p[rows-1-by-size:rows-1-by, bx:bx+size] = blob[::-1, :]
        return np.fft.ifft2(p**2).real, True

    def _gaussian_blob(self, size, intensity):
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        d = np.sqrt((x-size//2)**2 + (y-size//2)**2) / (size/2)
        return intensity * np.exp(-5 * d**2)

    def pattern_symmetric_grid(self, cols, rows):
        p = np.zeros((rows, cols))
        v = np.random.uniform(1, 50)
        y, x = np.ogrid[:rows, :cols]
        p = 128 + 127 * np.sin(2 * np.pi * v * x / cols) * np.sin(2 * np.pi * v * y / rows)
        return p, True

# ==========================================
# 2. MSCOCO DATASET WITH DCT AUGMENTATION
# ==========================================
class CocoFrequencyDataset(Dataset):
    def __init__(self, coco_folder, img_size=224, max_images=30000):
        print("--> Scanning MSCOCO folder for images (this might take a minute)...", flush=True)
        all_images = glob.glob(os.path.join(coco_folder, "*.jpg"))
        print(f"--> Found {len(all_images)} images on disk.", flush=True)
        
        if len(all_images) > max_images:
            print(f"--> Randomly selecting {max_images} images for training...", flush=True)
            random.seed(42)
            all_images = random.sample(all_images, max_images)
            
        self.image_paths = all_images
        self.img_size = img_size
        self.injector = FrequencyInjector()
        print("--> Dataset ready!", flush=True)

    def __len__(self):
        return len(self.image_paths)

    def _apply_dct(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dct = cv2.dct(cv2.dct(np.float32(gray), flags=cv2.DCT_ROWS), flags=cv2.DCT_ROWS)
        mask = np.log(np.abs(dct) + 1).astype(np.uint8)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        return cv2.merge((mask, mask, mask))

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Random augmentations BEFORE injection
        if random.random() < 0.5: img = cv2.flip(img, 1)
        if random.random() < 0.3: img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        if random.random() < 0.3:
            enc = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(40, 95)]
            _, e = cv2.imencode('.jpg', img, enc)
            img = cv2.imdecode(e, 1)
        
        # DCT Augmentation (Forces frequency learning)
        if random.random() < 0.3: img = self._apply_dct(img)
            
        # Inject pattern (50% chance)
        if random.random() < 0.5:
            img = self.injector.apply(img)
            label = 1.0
        else:
            label = 0.0
            
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1) 
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = normalize(img)
        return img, torch.tensor(label, dtype=torch.float32)

# ==========================================
# 3. FOCAL LOSS
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        return torch.mean(self.alpha * (1-pt)**self.gamma * bce)

# ==========================================
# 4. TRAINING LOOP (ResNet50, Plan A)
# ==========================================
def train():
    print("Starting Plan A training script...", flush=True)
    
    TRAIN_DIR = r"C:\Users\mda16\OneDrive\Desktop\DeepfakeDetector\paper1_benchmark_data\train2017"
    
    BATCH_SIZE = 4          
    ACCUM_STEPS = 8         # Effective batch = 32
    EPOCHS = 1
    LR = 0.0001

    print("Loading MSCOCO Data (30k subset)...", flush=True)
    dataset = CocoFrequencyDataset(TRAIN_DIR, img_size=224, max_images=30000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    print("Loading ResNet50...", flush=True)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.cuda()

    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    print(f"Plan A: ResNet50 on MSCOCO, batch {BATCH_SIZE} x accum {ACCUM_STEPS}", flush=True)
    
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
        
    torch.save(model.state_dict(), "verity_planA_resnet50.pth")
    print("Plan A Training Complete! Saved as verity_planA_resnet50.pth", flush=True)

if __name__ == "__main__":
    train()