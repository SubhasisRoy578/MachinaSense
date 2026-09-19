import os
import urllib.request
import zipfile
import shutil

RAW_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "ml", "data", "raw"))
ZIP_PATH = os.path.join(RAW_DIR, "CMAPSSData.zip")

NASA_ZIP_URL = "https://ti.arc.nasa.gov/c/6/"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/master/notebooks/jupyter/Dataset/CMAPSSData"

FD001_FILES = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]

def download_dataset():
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR, exist_ok=True)

    all_exist = all(os.path.exists(os.path.join(RAW_DIR, f)) for f in FD001_FILES)
    if all_exist:
        print(f"FD001 dataset files already present in {RAW_DIR}. Skipping download.")
        return

    print("Preparing C-MAPSS FD001 dataset...")
    
    # Try direct file download from raw GitHub mirror first as fallback-ready source
    try:
        print("Fetching C-MAPSS FD001 files from public dataset mirror...")
        for fname in FD001_FILES:
            target_path = os.path.join(RAW_DIR, fname)
            if not os.path.exists(target_path):
                file_url = f"{GITHUB_RAW_BASE}/{fname}"
                print(f"Downloading {fname} from {file_url}...")
                req = urllib.request.Request(file_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as resp, open(target_path, 'wb') as out_file:
                    shutil.copyfileobj(resp, out_file)
        print(f"C-MAPSS FD001 dataset successfully prepared in {RAW_DIR}")
        return
    except Exception as e:
        print(f"GitHub mirror download failed: {e}. Trying NASA zip download...")

    # Fallback to NASA zip download
    try:
        req = urllib.request.Request(NASA_ZIP_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(ZIP_PATH, 'wb') as out_file:
            shutil.copyfileobj(resp, out_file)
        
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
        
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
        print(f"Dataset successfully downloaded and extracted into {RAW_DIR}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\n--- MANUAL INSTRUCTIONS ---")
        print("Please place train_FD001.txt, test_FD001.txt, and RUL_FD001.txt in:")
        print(os.path.abspath(RAW_DIR))

if __name__ == "__main__":
    download_dataset()

