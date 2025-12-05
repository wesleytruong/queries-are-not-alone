#!/usr/bin/env python3
"""Download and unzip the MSRVTT clip archive from Hugging Face."""

import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "friedrichor/MSR-VTT"
ZIP_NAME = "MSRVTT_Videos.zip"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "msrvtt_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print(f"Downloading {ZIP_NAME} from {REPO_ID} ...")
    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=ZIP_NAME,
        revision="main",
        repo_type="dataset",
        local_dir=str(RAW_DIR),
    )

    target_dir = RAW_DIR
    target_dir.mkdir(exist_ok=True)
    print(f"Extracting into {target_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    print("Done.")


if __name__ == "__main__":
    main()
