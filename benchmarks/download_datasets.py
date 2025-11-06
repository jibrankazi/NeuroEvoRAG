# benchmarks/download_datasets.py
import os
import requests
from tqdm import tqdm

def download_file(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {os.path.basename(dest)}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    with open(dest, 'wb') as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest)
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

def download_hotpotqa():
    os.makedirs("benchmarks/datasets/hotpotqa", exist_ok=True)
    train_url = "https://huggingface.co/datasets/hotpotqa/resolve/main/hotpot_train_v1.1.json"
    dev_url = "https://huggingface.co/datasets/hotpotqa/resolve/main/hotpot_dev_distractor_v1.json"
    download_file(train_url, "benchmarks/datasets/hotpotqa/train.json")
    download_file(dev_url, "benchmarks/datasets/hotpotqa/dev.json")

def download_mmqa():
    print("MMQA: Skipping (requires torch) — use HotpotQA for now.")

def download_spokenhotpotqa():
    print("SpokenHotpotQA: Skipping (requires torch) — use HotpotQA.")

def download_legalbench():
    print("LegalBench: Skipping (requires torch) — use HotpotQA.")

if __name__ == "__main__":
    download_hotpotqa()
    print("\nHotpotQA downloaded. Other datasets skipped due to torch issue.")
    print("You can now run evolution on HotpotQA!")