"""
deploy_to_hf.py — Hosting script
Pushes all app files to the Hugging Face Space:
  GauthamJ007/agentic-feedback-analyzer

Usage:
    export HF_TOKEN=hf_...
    python hosting/deploy_to_hf.py
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SPACE_REPO_ID = "GauthamJ007/agentic-feedback-analyzer"
SPACE_SDK     = "docker"

# Repo root: one level up from this script (hosting/)
_script_dir = Path(__file__).resolve().parent
REPO_ROOT   = _script_dir.parent

# Files to push to HF Space (relative to REPO_ROOT)
FILES_TO_PUSH = [
    "app.py",
    "pipeline.py",
    "agents.py",
    "tasks.py",
    "Dockerfile",
    "requirements.txt",
    "README.md",
]

# Data files bundled into the Space image
DATA_FILES = [
    "data/app_store_reviews.csv",
    "data/support_emails.csv",
    "data/expected_classifications.csv",
]


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable is not set.")
        sys.exit(1)

    api = HfApi(token=token)

    # 1. Create (or verify) the Space
    try:
        api.repo_info(repo_id=SPACE_REPO_ID, repo_type="space")
        print(f"Space already exists: {SPACE_REPO_ID}")
    except RepositoryNotFoundError:
        create_repo(
            repo_id=SPACE_REPO_ID,
            repo_type="space",
            space_sdk=SPACE_SDK,
            private=False,
            token=token,
        )
        print(f"Created Space: {SPACE_REPO_ID} (sdk={SPACE_SDK})")

    # 2. Upload top-level app files
    for filename in FILES_TO_PUSH:
        local_path = REPO_ROOT / filename
        if not local_path.exists():
            print(f"WARN: {local_path} not found — skipping.")
            continue
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=SPACE_REPO_ID,
            repo_type="space",
            token=token,
        )
        print(f"Uploaded: {filename}")

    # 3. Upload bundled data files
    for rel_path in DATA_FILES:
        local_path = REPO_ROOT / rel_path
        if not local_path.exists():
            print(f"WARN: {local_path} not found — skipping.")
            continue
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=rel_path,
            repo_id=SPACE_REPO_ID,
            repo_type="space",
            token=token,
        )
        print(f"Uploaded: {rel_path}")

    print(f"\nAll files pushed to Hugging Face Space.")
    print(f"Space URL: https://huggingface.co/spaces/{SPACE_REPO_ID}")


if __name__ == "__main__":
    main()
