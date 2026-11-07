import pandas as pd

from torch.utils.data import DataLoader

from config import TRAIN_CSV, DATA_ROOT
from dataset import FaceDataset
from transforms import train_transform


# Load the training CSV
train_df = pd.read_csv(TRAIN_CSV)

# Build full image paths
train_df["full_path"] = train_df["path"].apply(
    lambda x: DATA_ROOT / x
)


# Create dataset
train_dataset = FaceDataset(
    dataframe=train_df,
    transform=train_transform
)


# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)


# Load one batch
images, labels = next(iter(train_loader))


print("Number of images in dataset:", len(train_dataset))
print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)
print("Labels:", labels)