"""RQ4 real-world false-alarm estimate with natural tracers in AI Village (local only, research terms).

Natural tracer: a rare token (long id, URL path, 5+ digit number, mixed-case slug) that appears in
exactly one agent's PRIVATE activity (computer-use tool outputs and actions) before it appears in
any public chat. A "transfer with no visible path" is a later public chat message by a DIFFERENT
agent containing that token, with no earlier public message by anyone containing it.
Such events are false alarms for a knowledge-closure audit of this chat log, because the real path
(a shared website, a shared document) was outside the log. The rate is reported per tracer and per
message. Tokens are hashed in the output so no content is quoted.
Usage: python scripts/ai_village_tracers.py --village swarm-village
"""
import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, RESULTS_DIR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--village", default="swarm-village")
ap.add_argument("--max-turns", type=int, default=400000)
ap.add_argument("--include-memories", action="store_true",
                help="also treat each agent's private memory notes (agent_memories) as private activity")
args = ap.parse_args()
base = DATA_DIR / "ai_village" / args.village

TOKEN_RE = re.compile(r"\b(?:[A-Za-z0-9]{6,}-[A-Za-z0-9-]{4,}|[a-z]+[A-Z][A-Za-z0-9]{6,}|\d{5,}|[a-f0-9]{12,}|[A-Za-z0-9_\-]{16,})\b")
STOP_PREFIXES = ("http", "www", "REDACTED", "BLOB", "IMAGE")


def read(name):
    p = base / name
    if not p.exists():
        return
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def tokens(text):
    out = set()
    for m in TOKEN_RE.findall(text or ""):
        if m.startswith(STOP_PREFIXES) or m.isdigit() and len(m) > 12:
            continue
        out.add(m)
    return out


agents = {a["id"]: a["name"] for a in read("agents.jsonl.gz")}
sessions = {s["id"]: s["agent_id"] for s in read("computer_use_sessions.jsonl.gz")}
print("agents:", len(agents), "sessions:", len(sessions))

# private activity: computer-use turns (actions + tool outputs), keyed by token -> (first_time, agent)
private_first = {}
private_agents = defaultdict(set)
n_turns = 0
for t in read("computer_use_turns.jsonl.gz"):
    n_turns += 1
    if n_turns > args.max_turns:
        break
    agent = sessions.get(t.get("session_id"))
    if not agent:
        continue
    text = json.dumps(t.get("agent_action") or {}) + " " + (t.get("output") or "")[:4000]
    ts = t.get("created_at", "")
    for tok in tokens(text):
        private_agents[tok].add(agent)
        if tok not in private_first or ts < private_first[tok][0]:
            private_first[tok] = (ts, agent)
n_mem = 0
if args.include_memories:
    for mrow in read("agent_memories.jsonl.gz"):
        agent = mrow.get("agent_id")
        if not agent:
            continue
        n_mem += 1
        ts = mrow.get("created_at", "")
        for tok in tokens(mrow.get("content") or ""):
            private_agents[tok].add(agent)
            if tok not in private_first or ts < private_first[tok][0]:
                private_first[tok] = (ts, agent)
print("turns scanned:", n_turns, "memories scanned:", n_mem, "candidate tokens:", len(private_first))

# public chat, in time order
chat = sorted((m for m in read("chat_messages.jsonl.gz") if m.get("speaker_type") == "agent"), key=lambda m: m["created_at"])
print("agent chat messages:", len(chat))
seen_public = {}            # token -> first public (time, speaker)
no_path, with_path, msgs_flagged = [], 0, set()
for m in chat:
    speaker, ts = m.get("agent_speaker_id"), m["created_at"]
    for tok in tokens(m.get("content")):
        if tok not in private_first or len(private_agents[tok]) != 1:
            continue  # not a tracer: never private, or private to several agents (shared source)
        owner_ts, owner = private_first[tok]
        if owner_ts >= ts:
            continue
        if tok in seen_public:
            with_path += 1
            continue
        seen_public[tok] = (ts, speaker)
        if speaker != owner:
            no_path.append({"token_sha": hashlib.sha256(tok.encode()).hexdigest()[:12], "owner": agents.get(owner, owner)[:30],
                            "speaker": agents.get(speaker, speaker)[:30], "time": ts, "message_id": m["id"]})
            msgs_flagged.add(m["id"])
tracers_total = sum(1 for tok, ags in private_agents.items() if len(ags) == 1)
summary = {"village": args.village, "agents": len(agents), "turns_scanned": n_turns, "memories_scanned": n_mem,
           "chat_messages": len(chat),
           "single_owner_tracers": tracers_total, "tracers_seen_in_chat_with_visible_path": with_path,
           "no_visible_path_events": len(no_path), "messages_flagged": len(msgs_flagged),
           "message_false_alarm_rate": round(len(msgs_flagged) / max(1, len(chat)), 5),
           "tracer_false_alarm_rate": round(len(no_path) / max(1, tracers_total), 5),
           "examples_hashed": no_path[:20]}
suffix = "_with_memories" if args.include_memories else ""
(RESULTS_DIR / f"ai_village_tracers_{args.village}{suffix}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "examples_hashed"}, indent=1))
