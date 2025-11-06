import os
from datasets import load_dataset
import requests
from tqdm import tqdm


def download_file(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192), desc=os.path.basename(dest)):
            f.write(chunk)


def download_hotpotqa():
    print("Downloading HotpotQA dataset via datasets library...")
    os.makedirs("benchmarks/datasets/hotpotqa", exist_ok=True)
    ds_train = load_dataset("hotpotqa", split="train")
    ds_train.save_to_disk("benchmarks/datasets/hotpotqa/train")
    ds_dev = load_dataset("hotpotqa", split="validation")
    ds_dev.save_to_disk("benchmarks/datasets/hotpotqa/dev")


def download_mmqa():
    print("MMQA requires cloning repo with images — using datasets library instead...")
    ds = load_dataset("MMQA/MMQA", split="test")
    ds.save_to_disk("benchmarks/datasets/mmqa")


def download_spokenhotpotqa():
    print("Downloading SpokenHotpotQA...")
    ds = load_dataset("the-bird-F/HotpotQA_RGBzh_speech")
    ds.save_to_disk("benchmarks/datasets/spokenhotpotqa")


def download_legalbench():
    print("Downloading LegalBench-RAG (contract_review)...")
    ds = load_dataset("lbox/lbox_open", "contract_review")
    ds.save_to_disk("benchmarks/datasets/legalbench")


if __name__ == "__main__":
    download_hotpotqa()
    download_mmqa()
    download_spokenhotpotqa()
    download_legalbench()
    print("All datasets downloaded to benchmarks/datasets/")
