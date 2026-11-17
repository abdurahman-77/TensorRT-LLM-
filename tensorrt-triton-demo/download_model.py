#!/usr/bin/env python3
"""Download HuggingFace model to persistent storage."""

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_DIR = Path("/persistent-storage/models") / MODEL_ID


def download_model():
    """Download the model, or finish a partial download, into persistent storage."""
    hf_token = os.environ.get("HF_AUTH_TOKEN")
    if not hf_token:
        sys.exit("ERROR: HF_AUTH_TOKEN is not set. Add it as a Cerebrium secret.")

    print(f"Syncing {MODEL_ID} into {MODEL_DIR} ...")
    # snapshot_download skips files that are already complete and resumes partial ones,
    # so an interrupted earlier run cannot leave a half-downloaded model behind.
    try:
        snapshot_download(MODEL_ID, local_dir=str(MODEL_DIR), token=hf_token)
    except GatedRepoError:
        sys.exit(
            f"ERROR: the token's account has no access to {MODEL_ID}. "
            f"Accept the license at https://huggingface.co/{MODEL_ID} and retry."
        )
    print("✓ Model ready")


if __name__ == "__main__":
    download_model()
