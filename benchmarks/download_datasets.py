import os
import sys
from datasets import load_dataset
import requests
from tqdm import tqdm

# This function is kept for non-HF downloads, but is NOT used by download_hotpotqa.
def download_file(url, dest):
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
    """
    Download the HotpotQA dataset using the robust datasets library method.
    This logic REPLACES the old, broken 'download_file' call.
    """
    print("--- CORRECT SCRIPT: Using 'load_dataset' for HotpotQA ---")
    print("Downloading HotpotQA (train/validation) via datasets library...")
    os.makedirs("benchmarks/datasets/hotpotqa", exist_ok=True)
    
    # Get the token from the environment variable (which you set with 'export')
    token = os.environ.get("HF_TOKEN")
    
    if not token:
        print("Warning: HF_TOKEN environment variable not set. Download may fail if repo is private/gated.", file=sys.stderr)

    try:
        # This robust approach handles authentication automatically via the 'token' argument
        train_ds = load_dataset(
            "hotpotqa/hotpot_qa",  # <-- CORRECTED PATH
            "distractor",
            split="train",
            token=token
        )
        train_ds.save_to_disk("benchmarks/datasets/hotpotqa/train")
        
        dev_ds = load_dataset(
            "hotpotqa/hotpot_qa",  # <-- CORRECTED PATH
            "distractor",
            split="validation",
            token=token
        )
        dev_ds.save_to_disk("benchmarks/datasets/hotpotqa/dev")
        print("--- HotpotQA Download Complete ---")

    except Exception as e:
        print(f"Error during load_dataset for HotpotQA: {e}", file=sys.stderr)
        print("Please ensure your HF_TOKEN is set correctly and you have accepted any terms on the dataset's Hugging Face page.", file=sys.stderr)
        sys.exit(1) # Exit script if this critical step fails


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
