import os
from datasets import load_dataset


def save_split(ds, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.save_to_disk(path)


def download_hotpotqa():
    """
    HotpotQA is public, so no token is needed here.
    """
    print("Downloading HotpotQA (distractor split)...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
    save_split(ds["train"], "benchmarks/datasets/hotpotqa/train")
    save_split(ds["validation"], "benchmarks/datasets/hotpotqa/dev")


def download_mmqa():
    """
    Use the public MMQA repo that currently exists on HF.
    """
    print("Downloading MMQA...")
    ds = load_dataset("TableQAKit/MMQA")
    for split in ds.keys():
        save_split(ds[split], f"benchmarks/datasets/mmqa/{split}")


def download_spokenhotpotqa():
    """
    Dataset exists on HF, but may still need special handling depending on its file layout.
    """
    print("Downloading SpokenHotpotQA...")
    ds = load_dataset("the-bird-F/HotpotQA_RGBzh_speech")
    for split in ds.keys():
        save_split(ds[split], f"benchmarks/datasets/spokenhotpotqa/{split}")


def download_legalbench():
    """
    'contract_review' is not a listed config for lbox/lbox_open.
    Pick one verified config instead, or skip this dataset for now.
    """
    print("Downloading lbox_open summarization...")
    ds = load_dataset("lbox/lbox_open", "summarization")
    for split in ds.keys():
        save_split(ds[split], f"benchmarks/datasets/legalbench/{split}")


if __name__ == "__main__":
    download_hotpotqa()
    download_mmqa()
    download_spokenhotpotqa()
    download_legalbench()
    print("All datasets downloaded to benchmarks/datasets/")
