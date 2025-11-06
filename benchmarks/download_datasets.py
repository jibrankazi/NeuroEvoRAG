import os
from datasets import load_dataset
import requests
from tqdm import tqdm


def download_file(url, dest):
    """
    Downloads a file from a given URL to a destination path, 
    using the HF_TOKEN environment variable for authorization if available.
    
    NOTE: This is retained for non-HuggingFace downloads, but load_dataset is 
    preferred for HF datasets.
    """
    print(f"Downloading {os.path.basename(dest)}...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # --- Token logic for manual requests ---
    headers = {}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        # Include the token in the Authorization header for manual requests
        headers["Authorization"] = f"Bearer {hf_token}"
    # --------------------------------------

    # Pass the headers to the requests.get call
    response = requests.get(url, headers=headers, stream=True)
    
    # This will raise an exception for 4xx or 5xx status codes
    response.raise_for_status()

    with open(dest, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192), desc=os.path.basename(dest)):
            f.write(chunk)


def download_hotpotqa():
    # NOTICE: This print statement is unique. If you see this, the file is updated.
    print("Downloading HotpotQA dataset via datasets library (uses saved HF token automatically)...")
    os.makedirs("benchmarks/datasets/hotpotqa", exist_ok=True)
    
    # --- THIS CORRECTLY USES load_dataset TO AVOID THE 401 ERROR ---
    ds_train = load_dataset("hotpotqa", split="train")
    ds_train.save_to_disk("benchmarks/datasets/hotpotqa/train")
    ds_dev = load_dataset("hotpotqa", split="validation")
    ds_dev.save_to_disk("benchmarks/datasets/hotpotqa/dev")
    # -----------------------------------------------------


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
