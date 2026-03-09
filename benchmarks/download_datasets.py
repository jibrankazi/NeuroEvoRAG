import os
from datasets import load_dataset
from huggingface_hub import snapshot_download


def save_split(ds, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.save_to_disk(path)


def hf_load_dataset(*args, **kwargs):
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        kwargs["token"] = hf_token
    return load_dataset(*args, **kwargs)


def download_hotpotqa():
    print("Downloading HotpotQA (distractor split)...")
    ds = hf_load_dataset("hotpotqa/hotpot_qa", "distractor")
    save_split(ds["train"], "benchmarks/datasets/hotpotqa/train")
    save_split(ds["validation"], "benchmarks/datasets/hotpotqa/dev")


def download_mmqa():
    print("Downloading MMQA raw files...")
    os.makedirs("benchmarks/datasets/mmqa", exist_ok=True)

    snapshot_download(
        repo_id="TableQAKit/MMQA",
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
        local_dir="benchmarks/datasets/mmqa",
        allow_patterns=[
            "MMQA_train.jsonl.gz",
            "MMQA_dev.jsonl.gz",
            "MMQA_test.jsonl.gz",
            "MMQA_texts.jsonl.gz",
            "MMQA_tables.jsonl.gz",
            "MMQA_images.jsonl.gz",
            "mmqa.py",
        ],
    )


def download_spokenhotpotqa():
    print("Downloading SpokenHotpotQA...")
    ds = hf_load_dataset("the-bird-F/HotpotQA_RGBzh_speech")
    for split in ds.keys():
        save_split(ds[split], f"benchmarks/datasets/spokenhotpotqa/{split}")


def download_legalbench():
    print("Downloading LegalBench...")
    ds = hf_load_dataset("lbox/lbox_open", "summarization")
    for split in ds.keys():
        save_split(ds[split], f"benchmarks/datasets/legalbench/{split}")


if __name__ == "__main__":
    download_hotpotqa()
    download_mmqa()
    download_spokenhotpotqa()
    download_legalbench()
    print("All datasets downloaded to benchmarks/datasets/")
