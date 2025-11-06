import os
from datasets import load_dataset
import requests
from tqdm import tqdm

# Note: The manual download_file function is kept but is NOT used for HotpotQA,
# as the load_dataset function (below) is required for proper authentication.
def download_file(url, dest):
    """Placeholder for manual non-HF downloads."""
    print(f"Executing manual download of {os.path.basename(dest)}...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    headers = {}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()

    with open(dest, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192), desc=os.path.basename(dest)):
            f.write(chunk)


def download_hotpotqa():
    """Download the HotpotQA dataset using the robust datasets library method."""
    print("--- SUCCESS: Using Fixed load_dataset logic for HotpotQA ---")
    print("Downloading HotpotQA (train/validation) via datasets library...")
    os.makedirs("benchmarks/datasets/hotpotqa", exist_ok=True)
    
    # Get the token from the environment variable (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)
    token = os.environ.get("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")
    
    # This robust approach handles authentication automatically via the 'token' argument
    train_ds = load_dataset(
        "hotpotqa", 
        "distractor",
        split="train",
        token=token
    )
    train_ds.save_to_disk("benchmarks/datasets/hotpotqa/train")
    
    dev_ds = load_dataset(
        "hotpotqa", 
        "distractor",
        split="validation",
        token=token
    )
    dev_ds.save_to_disk("benchmarks/datasets/hotpotqa/dev")


def download_mmqa():
    print("Downloading MMQA...")
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
