import os
from datasets import load_dataset

def download_hotpotqa():
    print("Downloading HotpotQA (distractor split) via datasets library...")
    os.makedirs("benchmarks/datasets/hotpotqa", exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")

    train_ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="train",
        use_auth_token=hf_token
    )
    train_ds.save_to_disk("benchmarks/datasets/hotpotqa/train")

    dev_ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
        use_auth_token=hf_token
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
