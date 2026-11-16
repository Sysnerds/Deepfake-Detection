import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# ==========================================
# 1. TEST DATASET FROM CSV
# ==========================================
class BenchmarkDataset(Dataset):
    def __init__(self, csv_file, img_size=224):
        self.df = pd.read_csv(csv_file)
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        img = self.transform(img)
        return img, row['label'], row['method']

# ==========================================
# 2. EVALUATION
# ==========================================
def evaluate():
    CSV_PATH = r"C:\Users\mda16\OneDrive\Desktop\DeepfakeDetector\paper1_benchmark_data\paper1_face_benchmark.csv"
    
    # UPDATE THIS PATH TO WHERE YOUR MODEL WAS SAVED
    MODEL_PATH = r"C:\Users\mda16\OneDrive\Desktop\DeepfakeDetector\verity_planA_resnet50.pth"

    print("Loading Benchmark Data...")
    dataset = BenchmarkDataset(CSV_PATH, img_size=224)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    print("Loading Plan A ResNet50 Model...")
    # Initialize ResNet50
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    
    # Load the weights (added map_location for safety)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location='cuda')) 
    model = model.cuda()
    model.eval()

    all_preds, all_labels, all_methods = [], [], []

    print("Running inference...")
    with torch.no_grad():
        for images, labels, methods in dataloader:
            images = images.cuda()
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int).flatten()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_methods.extend(methods)

    # ==========================================
    # 3. OVERALL METRICS
    # ==========================================
    print("\n" + "="*50)
    print("OVERALL BENCHMARK RESULTS (Plan A ResNet50)")
    print("="*50)
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"                 Predicted Real   Predicted Fake")
    print(f"Actual Real      {cm[0][0]:<16} {cm[0][1]:<16}")
    print(f"Actual Fake      {cm[1][0]:<16} {cm[1][1]:<16}")

    # ==========================================
    # 4. PER-GENERATOR METRICS
    # ==========================================
    print("\n" + "="*50)
    print("PER-GENERATOR RECALL (The Paper 1 Metric)")
    print("="*50)
    
    results_df = pd.DataFrame({'label': all_labels, 'pred': all_preds, 'method': all_methods})
    
    for method in results_df['method'].unique():
        subset = results_df[results_df['method'] == method]
        correct = (subset['label'] == subset['pred']).sum()
        total = len(subset)
        
        if subset['label'].iloc[0] == 1: 
            recall = correct / total
            print(f"{method:<20}: Recall = {recall * 100:.2f}% ({correct}/{total})")
        else: 
            specificity = correct / total
            print(f"{method:<20}: Specificity = {specificity * 100:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    evaluate()