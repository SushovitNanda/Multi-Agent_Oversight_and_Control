"""Step 1: list what this account can actually call. Saves results/models.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR, adaption_client  # noqa: E402

client = adaption_client()


def dump(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else dict(obj)


hosted = [dump(m) for m in client.models.list(type="hosted").data]
everything = [dump(m) for m in client.models.list().data]
training = []
try:
    training = [dump(m) for m in client.autoscientist.list_models().models]
except Exception as exc:  # noqa: BLE001
    print("autoscientist.list_models failed:", type(exc).__name__, str(exc)[:200])

(RESULTS_DIR / "models.json").write_text(json.dumps(
    {"hosted": hosted, "all": everything, "training": training}, indent=2, default=str), encoding="utf-8")

print("HOSTED, callable with responses.create")
for m in hosted:
    print(" ", m.get("id"), "| shutdown:", m.get("shutdown_date"))
print("ALL models:", len(everything))
print("TRAINING BASE MODELS, for AutoScientist")
for m in training:
    print(" ", m.get("id"), "|", m.get("model_size"), "|", m.get("training_types"))
