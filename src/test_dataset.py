from pathlib import Path

import pandas as pd

from config import TRAIN_CSV, DATA_ROOT
from dataset import FaceDataset
from transforms import train_transform


# Load the training CSV
train_df = pd.read_csv(TRAIN_CSV)

# Build the actual image path
train_df["full_path"] = train_df["path"].apply(
    lambda x: DATA_ROOT / x
)


# Create the dataset
dataset = FaceDataset(
    dataframe=train_df,
    transform=train_transform
)


# Load one image
image, label = dataset[0]


print("Dataset size:", len(dataset))
print("Image tensor shape:", image.shape)
print("Label:", label)
print("Image path:", train_df.iloc[0]["full_path"])