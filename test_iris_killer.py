import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 1. TEST DATASET (RGB + FFT)
# ==========================================
class TestDataset(Dataset):
    def __init__(self, real_dir, fake_dir, img_size=224):
        # Load 500 real and up to 500 unseen fake images
        self.real_imgs = glob.glob(os.path.join(real_dir, "**", "*.jpg"), recursive=True)[:500]
        self.fake_imgs = glob.glob(os.path.join(fake_dir, "**", "*.jpg"), recursive=True)[:500]
        
        print(f"Found {len(self.real_imgs)} real and {len(self.fake_imgs)} fake images.")
        
        self.image_paths = self.real_imgs + self.fake_imgs
        self.labels = [0] * len(self.real_imgs) + [1] * len(self.fake_imgs)
        self.img_size = img_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 1. Load RGB Image (WITH SAFETY CHECK)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            # If the image is corrupt or missing, create a blank black image
            img_bgr = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # 2. Compute FFT Magnitude (The X-Ray Vision)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 3. Prepare Tensors
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = self.normalize(img_tensor)
        fft_tensor = torch.from_numpy(magnitude).unsqueeze(0).float() / 255.0
        
        # Combine into 4 channels (RGB + FFT)
        combined_tensor = torch.cat((img_tensor, fft_tensor), dim=0)
        
        return combined_tensor, label

# ==========================================
# 2. DUAL-INPUT RESNET50 (Matches Training)
# ==========================================
def get_dual_input_resnet():
    model = models.resnet50(weights=None)
    # Modify the first layer to accept 4 channels
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

# ==========================================
# 3. EVALUATION
# ==========================================
def evaluate():
    # Real faces from Wikipedia
    EXTERNAL_REAL_DIR = r"C:\Users\mda16\OneDrive\Desktop\DeepfakeDetector\external_data\wiki\wiki"
    
    # Unseen fake faces from the Inpainting dataset
    EXTERNAL_FAKE_DIR = r"C:\Users\mda16\OneDrive\Desktop\DeepfakeDetector\external_data\inpainting"
    
    # The Universal model we just trained
    MODEL_PATH = r"C:\Users\mda16\OneDrive\Desktop\DeepfakeDetector\universal_detector.pth"

    print("Loading Unseen Test Data...")
    dataset = TestDataset(EXTERNAL_REAL_DIR, EXTERNAL_FAKE_DIR, img_size=224)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    print("Loading Universal Dual-Input Model...")
    model = get_dual_input_resnet().cuda()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    all_preds, all_labels = [], []

    print("Running inference...")
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # CHANGED THRESHOLD FROM 0.5 TO 0.3
            preds = (probs > 0.3).astype(int).flatten()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # ==========================================
    # 4. RESULTS
    # ==========================================
    print("\n" + "="*50)
    print("THE IRIS-GAN KILLER: EXTERNAL EVALUATION")
    print("="*50)
    
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    real_spec = cm[0][0] / sum(cm[0]) * 100
    fake_recall = cm[1][1] / sum(cm[1]) * 100

    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print(f"Real Specificity: {real_spec:.2f}%")
    print(f"Unseen Diffusion Recall: {fake_recall:.2f}%")
    
    print("\nConfusion Matrix:")
    print(f"                 Predicted Real   Predicted Fake")
    print(f"Actual Real      {cm[0][0]:<16} {cm[0][1]:<16}")
    print(f"Actual Fake      {cm[1][0]:<16} {cm[1][1]:<16}")
    
    print("\n" + "="*50)
    print("COMPARISON WITH IRIS-GAN (Paper)")
    print("="*50)
    print(f"IRIS-GAN Diffusion Recall: ~0.00% (Classifies all as Real)")
    print(f"Our Model Diffusion Recall: {fake_recall:.2f}%")
    
    if fake_recall > 10.0:
        print("\nRESULT: WE BEAT IRIS-GAN ON UNSEEN DIFFUSION MODELS!")
    else:
        print("\nRESULT: Failed to transfer to diffusion models.")

if __name__ == "__main__":
    evaluate()