"""Step 2a: inspect the AI Village dataset before downloading anything large."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402,F401  (sets HF_HOME to D:\APART Downloads\hf_cache)
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

REPO = "aidigestorg/ai-village"
api = HfApi()
rows = []
for item in api.list_repo_tree(REPO, repo_type="dataset", recursive=True):
    size = getattr(item, "size", None)
    if size is not None:
        rows.append((size, item.path))
rows.sort(reverse=True)

out = config.RESULTS_DIR / "ai_village_files.txt"
with out.open("w", encoding="utf-8") as f:
    for size, path in rows:
        f.write(f"{size / 1e9:9.3f} GB  {path}\n")
print("files:", len(rows), "total GB:", round(sum(s for s, _ in rows) / 1e9, 1), "->", out)
print("smallest 25 files (candidates for download):")
for size, path in sorted(rows)[:25]:
    print(f"  {size / 1e6:9.2f} MB  {path}")

for name in ("SCHEMA.md", "README.md"):
    try:
        local = hf_hub_download(REPO, name, repo_type="dataset", cache_dir=str(config.HF_CACHE))
        print("downloaded", name, "->", local)
    except Exception as exc:  # noqa: BLE001
        print("could not fetch", name, type(exc).__name__, str(exc)[:200])
