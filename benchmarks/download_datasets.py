import os
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import snapshot_download


BASE_DIR = "benchmarks/datasets"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_split(ds: Dataset, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    ds.save_to_disk(path)


def save_dataset_splits(ds, out_dir: str) -> None:
    ensure_dir(out_dir)

    if isinstance(ds, DatasetDict):
        for split_name, split_ds in ds.items():
            save_split(split_ds, os.path.join(out_dir, split_name))
    elif isinstance(ds, Dataset):
        save_split(ds, out_dir)
    else:
        raise TypeError(f"Unsupported dataset type: {type(ds)}")


def hf_load_dataset(*args, **kwargs):
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and "token" not in kwargs:
        kwargs["token"] = hf_token
    return load_dataset(*args, **kwargs)


def download_hotpotqa() -> None:
    print("Downloading HotpotQA (distractor split)...")
    ds = hf_load_dataset("hotpotqa/hotpot_qa", "distractor")
    save_split(ds["train"], os.path.join(BASE_DIR, "hotpotqa", "train"))
    save_split(ds["validation"], os.path.join(BASE_DIR, "hotpotqa", "dev"))


def download_mmqa() -> None:
    print("Downloading MMQA raw files...")
    out_dir = os.path.join(BASE_DIR, "mmqa")
    ensure_dir(out_dir)

    snapshot_download(
        repo_id="TableQAKit/MMQA",
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
        local_dir=out_dir,
        allow_patterns=[
            "MMQA*.jsonl.gz",
            "retriever_files/*",
            "mmqa.py",
            "README.md",
        ],
    )


def download_spokenhotpotqa() -> None:
    print("Downloading SpokenHotpotQA...")
    ds = hf_load_dataset("the-bird-F/HotpotQA_RGBzh_speech")
    save_dataset_splits(ds, os.path.join(BASE_DIR, "spokenhotpotqa"))


def download_legalbench() -> None:
    print("Downloading LegalBench (lbox_open/summarization)...")
    ds = hf_load_dataset("lbox/lbox_open", "summarization")
    save_dataset_splits(ds, os.path.join(BASE_DIR, "legalbench"))


def main() -> None:
    tasks = [
        ("hotpotqa", download_hotpotqa),
        ("mmqa", download_mmqa),
        ("spokenhotpotqa", download_spokenhotpotqa),
        ("legalbench", download_legalbench),
    ]

    failures = []

    for name, fn in tasks:
        try:
            fn()
            print(f"[OK] {name}")
        except Exception as e:
            failures.append((name, str(e)))
            print(f"[FAIL] {name}: {e}")

    print("\nDownload summary:")
    if failures:
        for name, err in failures:
            print(f" - {name}: FAILED -> {err}")
        raise SystemExit(1)

    print("All datasets downloaded to benchmarks/datasets/")


if __name__ == "__main__":
    main()
