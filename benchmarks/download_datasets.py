# benchmarks/download_datasets.py
import os
from datasets import load_dataset
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
    print("Downloading MMQA via Hugging Face datasets...")
    ds = load_dataset("MMQA/MMQA", split="test")
    ds.save_to_disk("benchmarks/datasets/mmqa")

def download_spokenhotpotqa():
    print("Downloading SpokenHotpotQA...")
    ds = load_dataset("the-bird-F/HotpotQA_RGBzh_speech")
    ds.save_to_disk("benchmarks/datasets/spokenhotpotqa")

def download_legalbench():
    print("Downloading LegalBench (contract_review)...")
    ds = load_dataset("lbox/lbox_open", "contract_review")
    ds.save_to_disk("benchmarks/datasets/legalbench")

if __name__ == "__main__":
    download_hotpotqa()
    download_mmqa()
    download_spokenhotpotqa()
    download_legalbench()
    print("\nAll datasets downloaded to benchmarks/datasets/")