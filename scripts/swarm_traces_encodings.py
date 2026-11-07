"""Survey encoding patterns in Swarm Traces to choose realistic decoder chains for RQ3.

Reports only aggregate counts by pattern. No payload text is printed or saved.
"""
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, RESULTS_DIR  # noqa: E402

src = DATA_DIR / "swarm_traces" / "redacted.jsonl.gz"
PATTERNS = {
    "base64_blob_32+": re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{32,}={0,2}(?![A-Za-z0-9+/])"),
    "hex_blob_32+": re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32,}(?![0-9a-fA-F])"),
    "url_encoded": re.compile(r"(?:%[0-9A-Fa-f]{2}){3,}"),
    "atob_or_b64decode": re.compile(r"(?i)\batob\(|base64\.b64decode|from_base64|Buffer\.from\([^)]*base64"),
    "rot13_or_charcode": re.compile(r"(?i)rot13|fromCharCode|charCodeAt"),
    "reverse_call": re.compile(r"(?i)\.reverse\(\)|\[::-1\]"),
    "hex_escape_seq": re.compile(r"(?:\\x[0-9a-fA-F]{2}){4,}"),
    "unicode_escape_seq": re.compile(r"(?:\\u[0-9a-fA-F]{4}){4,}"),
    "gzip_or_zlib": re.compile(r"(?i)\bgzip\b|zlib|inflate|decompress"),
    "eval_or_exec": re.compile(r"(?i)\beval\(|\bexec\(|new Function\("),
    "dir_name_message": re.compile(r"(?i)mkdir|os\.makedirs|cache"),
}
kinds, tags = Counter(), Counter()
hits, n = Counter(), 0
combo = Counter()
with gzip.open(src, "rt", encoding="utf-8") as f:
    for line in f:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        n += 1
        kinds[row.get("kind")] += 1
        for t in str(row.get("tags", "")).split(","):
            if t.strip():
                tags[t.strip()] += 1
        text = row.get("text") or ""
        present = [name for name, rx in PATTERNS.items() if rx.search(text)]
        for p in present:
            hits[p] += 1
        if len(present) >= 2:
            combo[tuple(sorted(present))] += 1
out = {"rows": n, "kinds": dict(kinds), "top_tags": tags.most_common(25),
       "pattern_rows": dict(hits), "pattern_rate": {k: round(v / n, 4) for k, v in hits.items()},
       "top_combinations": [{"patterns": list(k), "rows": v} for k, v in combo.most_common(15)]}
(RESULTS_DIR / "swarm_traces_encodings.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
