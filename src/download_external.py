from huggingface_hub import snapshot_download
import os

print("Downloading DeepFakeFace dataset (this may take a minute)...")
# This downloads the dataset directly to a folder in your current directory
dataset_path = snapshot_download(
    repo_id="desingh/DeepFakeFace", 
    repo_type="dataset",
    local_dir="./external_data"
)

print(f"Downloaded successfully to: {dataset_path}")
print("Checking files...")
for root, dirs, files in os.walk("./external_data"):
    level = root.replace("./external_data", '').count(os.sep)
    indent = ' ' * 4 * (level)
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 1)
    # Only print a few files so it doesn't flood the terminal
    for f in files[:3]: 
        print(f"{subindent}{f}")
    if len(files) > 3:
        print(f"{subindent}... and {len(files) - 3} more files")