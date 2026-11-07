"""Step 2a: download only the small AI Village tables (no screenshots, no turn archives) to D:.

swarm-village is the smallest complete village (events 8 MB, memories 26 MB) and the most relevant one.
From main-village only the chat tables and summaries are taken. Research terms: local use only,
never uploaded to Adaption, never used for training.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "aidigestorg/ai-village"
FILES = [
    "swarm-village/agents.jsonl.gz", "swarm-village/chat_rooms.jsonl.gz", "swarm-village/chat_messages.jsonl.gz",
    "swarm-village/events.jsonl.gz", "swarm-village/agent_memories.jsonl.gz",
    "swarm-village/computer_use_sessions.jsonl.gz", "swarm-village/computer_use_turns.jsonl.gz",
    "swarm-village/villages.jsonl.gz", "swarm-village/manifest.json",
    "main-village/agents.jsonl.gz", "main-village/chat_rooms.jsonl.gz", "main-village/chat_messages.jsonl.gz",
    "main-village/summaries.jsonl.gz", "main-village/villages.jsonl.gz",
    "CHANGELOG.md",
]
dest = config.DATA_DIR / "ai_village"
dest.mkdir(exist_ok=True)
for f in FILES:
    try:
        local = hf_hub_download(REPO, f, repo_type="dataset", cache_dir=str(config.HF_CACHE),
                                local_dir=str(dest))
        print("ok", f, round(Path(local).stat().st_size / 1e6, 1), "MB")
    except Exception as exc:  # noqa: BLE001
        print("FAILED", f, type(exc).__name__, str(exc)[:160])
print("->", dest)
