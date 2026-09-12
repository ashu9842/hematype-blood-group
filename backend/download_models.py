import os
from huggingface_hub import hf_hub_download

REPO_ID = "ashu9842/hematype8-models"
FILES = ["model.pt", "fingerprint_validator.pt", "smear_validator.pt"]

for filename in FILES:
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=".",
        )
        print(f"{filename} done!")
    else:
        print(f"{filename} already exists, skipping.")