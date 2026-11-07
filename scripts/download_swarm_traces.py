"""Step 2b: download Swarm Traces (16 MB) to D:\\APART Downloads and summarise its fields. Local only."""
import gzip
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR  # noqa: E402

URL = "https://swarmtraces.org/data/final/redacted.jsonl.gz"
dest_dir = DATA_DIR / "swarm_traces"
dest_dir.mkdir(exist_ok=True)
dest = dest_dir / "redacted.jsonl.gz"
if not dest.exists():
    print("downloading", URL)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (research download; Apart sprint)"})
    with urllib.request.urlopen(req, timeout=120) as r, dest.open("wb") as f:
        f.write(r.read())
print("size MB:", round(dest.stat().st_size / 1e6, 1))

keys, n = Counter(), 0
sample = None
with gzip.open(dest, "rt", encoding="utf-8") as f:
    for line in f:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        n += 1
        keys.update(row.keys())
        if sample is None:
            sample = {k: (str(v)[:60] + "...") if len(str(v)) > 60 else v for k, v in row.items()}
print("rows:", n)
print("fields:", dict(keys))
print("sample (truncated, no payload quoted in full):", json.dumps(sample, indent=1)[:1500])
