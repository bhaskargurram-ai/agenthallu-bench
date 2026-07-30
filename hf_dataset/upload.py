"""Stage and upload the AgentProp-Bench dataset to the Hugging Face Hub.

Copies the three released artifacts into hf_dataset/ under the names the card's
`configs:` block expects, then (with --push) uploads the folder to a HF dataset
repo. A DOI is minted from the HF dataset settings page ("Generate DOI") once the
repo is live — add it to CITATION.cff and the README afterwards.

Usage:
  ./venv/bin/python hf_dataset/upload.py                      # stage files only (dry run)
  ./venv/bin/python hf_dataset/upload.py --push --repo user/AgentProp-Bench
Requires: `huggingface-cli login` (a token with write access).
"""
import argparse
import os
import shutil

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")

# source in repo  ->  name referenced by the dataset card's configs
FILES = {
    os.path.join(ROOT, "data/tasks/full_2000_tasks.json"): "tasks.json",
    os.path.join(ROOT, "data/results/p2_all_rejudged.csv"): "traces.csv",
    os.path.join(ROOT, "data/results/human_labeling_sheet_Yoshitha_labeled.csv"): "human_labels.csv",
}


def stage():
    for src, dst in FILES.items():
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        shutil.copyfile(src, os.path.join(HERE, dst))
        print(f"staged {dst}  <-  {os.path.relpath(src, ROOT)}")
    print("README.md (dataset card) already present." if os.path.exists(os.path.join(HERE, "README.md"))
          else "WARNING: README.md missing")


def push(repo):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=HERE, repo_id=repo, repo_type="dataset",
                      ignore_patterns=["upload.py", "__pycache__/*"])
    print(f"\npushed -> https://huggingface.co/datasets/{repo}")
    print("Next: open the dataset's Settings -> 'Generate DOI', then paste the DOI "
          "into CITATION.cff and README.md.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--push", action="store_true")
    p.add_argument("--repo", default="bhaskargurram/AgentProp-Bench")
    a = p.parse_args()
    stage()
    if a.push:
        push(a.repo)
    else:
        print("\n(dry run — re-run with --push --repo <user>/AgentProp-Bench to upload)")
