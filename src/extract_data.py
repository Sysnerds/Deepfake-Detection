import zipfile
import os

base_path = "./external_data"
zips_to_extract = ["wiki.zip", "text2img.zip"]

for zip_name in zips_to_extract:
    zip_path = os.path.join(base_path, zip_name)
    extract_folder = os.path.join(base_path, zip_name.replace(".zip", ""))
    
    if os.path.exists(zip_path):
        print(f"Extracting {zip_name} to {extract_folder}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        print(f"Done extracting {zip_name}.")
    else:
        print(f"Could not find {zip_name}")

# Now let's print the structure so we know where the images are
print("\n--- Directory Structure of Extracted Data ---")
for root, dirs, files in os.walk(base_path):
    # Skip the cache folder and the zip files themselves
    if ".cache" in root or root.endswith(base_path): 
        continue
    
    # Print folder name
    level = root.replace(base_path, '').count(os.sep)
    indent = ' ' * 4 * (level)
    print(f"{indent}{os.path.basename(root)}/")
    
    # Print up to 3 files per folder
    subindent = ' ' * 4 * (level + 1)
    for f in files[:3]:
        print(f"{subindent}{f}")
    if len(files) > 3:
        print(f"{subindent}... and {len(files) - 3} more files")