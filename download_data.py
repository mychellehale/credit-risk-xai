import kagglehub
import shutil
import os

# --- Configuration ---
DATASET_SLUG = "varunchawla30/german-credit-data"
EXPECTED_FILE_NAME = "german_credit_data.csv" # Confirm the name after checking the dataset page
# Define the target location within your professional project structure
TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')
TARGET_PATH = os.path.join(TARGET_DIR, EXPECTED_FILE_NAME)

if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

print(f"Downloading {DATASET_SLUG}...")

# 1. Download the dataset to the local Kaggle cache
try:
    cache_path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"Dataset downloaded to cache: {cache_path}")

    # 2. Find the actual file within the cache path
    source_file = os.path.join(cache_path, EXPECTED_FILE_NAME)

    # 3. Move the file from the cache to the project's data/raw folder
    print(f"Moving file to target directory: {TARGET_PATH}")
    shutil.move(source_file, TARGET_PATH)
    
    print("\n✅ Data acquisition complete!")
    print(f"The raw file is now located at: {TARGET_PATH}")

except Exception as e:
    print(f"\n❌ ERROR during data download or movement: {e}")
    print("Ensure you have run 'pip install kagglehub' and have your 'kaggle.json' set up.")