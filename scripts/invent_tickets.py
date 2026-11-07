"""Step 4: create the benign ticket corpus with Invent a Dataset. Estimate first; --run spends credits."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, RESULTS_DIR, adaption_client  # noqa: E402

ROWS = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 600
client = adaption_client()
PROMPT = """Generate realistic, benign software engineering tickets for a small team maintaining a
Python package called quillpack at the fictional company Ferrowave Systems. Each prompt is one
ticket: a short title on the first line, then "Component: <name>", then the task in two or three
sentences. Each completion is a concise plan of exactly three numbered steps, one line each.
Cover dependency updates, failing tests, config changes, documentation, and small bug fixes.
Use only fictional company, package, and host names. Do not include security exploits,
credentials, or real personal data."""

estimate = client.datasets.invent(dataset_prompt=PROMPT, rows=ROWS, name="benign-tickets", estimate=True)
print("rows:", ROWS, "estimated credits:", estimate.estimated_credits, "available:", estimate.available_credits)
if "--run" not in sys.argv:
    raise SystemExit("Estimate only. Re-run with --run to spend credits.")

dataset = client.datasets.invent(dataset_prompt=PROMPT, rows=ROWS, name="benign-tickets",
                                 idempotency_key=f"benign-tickets-v1-{ROWS}")
print("dataset id:", dataset.id, "status:", dataset.status)
(RESULTS_DIR / "invent_dataset_id.txt").write_text(dataset.id, encoding="utf-8")
while True:
    current = client.datasets.get(dataset.id)
    print(time.strftime("%H:%M:%S"), "status:", current.status)
    if current.status == "succeeded":
        break
    if current.status == "failed":
        err = getattr(current, "error_data", None)
        raise SystemExit(f"failed: {getattr(err, 'message', err)}")
    time.sleep(10)

out = DATA_DIR / "tickets.jsonl"
client.datasets.download(dataset.id, file_format="jsonl").write_to_file(str(out))
rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
print("downloaded rows:", len(rows), "->", out)
print("first row keys:", list(rows[0].keys()) if rows else None)
print(json.dumps(rows[0], indent=1)[:1200] if rows else "")
