import os
import subprocess


def download_hotpotqa():
    os.makedirs("benchmarks/datasets/hotpotqa", exist_ok=True)
    subprocess.run([
        "wget", "https://huggingface.co/datasets/hotpotqa/resolve/main/hotpot_train_v1.1.json",
        "-O", "benchmarks/datasets/hotpotqa/train.json"
    ])
    subprocess.run([
        "wget", "https://huggingface.co/datasets/hotpotqa/resolve/main/hotpot_dev_distractor_v1.json",
        "-O", "benchmarks/datasets/hotpotqa/dev.json"
    ])


def download_mmqa():
    os.makedirs("benchmarks/datasets/mmqa", exist_ok=True)
    subprocess.run([
        "git", "clone", "https://github.com/MMQA/MMQA.git",
        "benchmarks/datasets/mmqa"
    ])


def download_spokenhotpotqa():
    # Install datasets package and load SpokenHotpotQA from Hugging Face
    subprocess.run([
        "pip", "install", "datasets"
    ])
    from datasets import load_dataset
    ds = load_dataset("the-bird-F/HotpotQA_RGBzh_speech")
    ds.save_to_disk("benchmarks/datasets/spokenhotpotqa")


def download_legalbench():
    # Install datasets package and load LegalBench dataset
    subprocess.run([
        "pip", "install", "datasets"
    ])
    from datasets import load_dataset
    ds = load_dataset("lbox/lbox_open", "contract_review")
    ds.save_to_disk("benchmarks/datasets/legalbench")


if __name__ == "__main__":
    download_hotpotqa()
    download_mmqa()
    download_spokenhotpotqa()
    download_legalbench()
    print("All datasets downloaded to benchmarks/datasets/")
