"""Step 3: smoke test tool calling, parallel calls, and credit deductions for one model."""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR, adaption_client  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "deepseek-v4-flash"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 96
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 16
client = adaption_client()

TOOLS = [{"type": "function", "name": "read_ticket", "description": "Read a ticket by id.",
          "parameters": {"type": "object", "properties": {"ticket_id": {"type": "string"}},
                         "required": ["ticket_id"]}}]


def balance():
    try:
        est = client.datasets.invent(dataset_prompt="Short IT support tickets.", rows=1000, estimate=True)
        return est.available_credits
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({type(exc).__name__}: {str(exc)[:120]})"


def tool_round_trip():
    history = [{"role": "user", "content": "Read ticket T-1 and summarise it in one line."}]
    first = client.responses.create(model=MODEL, input=history, tools=TOOLS, tool_choice="auto",
                                    agentic_memory=False, store=False, max_output_tokens=200)
    calls = [item for item in first.output if item.type == "function_call"]
    if not calls:
        return first.usage, None, first.output_text
    call = calls[0]
    history.append({"type": "function_call", "call_id": call.call_id, "name": call.name,
                    "arguments": call.arguments})
    history.append({"type": "function_call_output", "call_id": call.call_id,
                    "output": json.dumps({"text": "Disk full on build host."})})
    second = client.responses.create(model=MODEL, input=history, tools=TOOLS, agentic_memory=False,
                                     store=False, max_output_tokens=200)
    return first.usage, call.name, second.output_text


def one_call(i):
    start = time.time()
    try:
        r = client.responses.create(model=MODEL, input=f"Reply with the number {i}.", agentic_memory=False,
                                    store=False, max_output_tokens=20)
        return True, time.time() - start, r.usage.input_tokens, r.usage.output_tokens, ""
    except Exception as exc:  # noqa: BLE001
        return False, time.time() - start, 0, 0, f"{type(exc).__name__}: {str(exc)[:160]}"


before = balance()
t0 = time.time()
usage, tool_name, text = tool_round_trip()
print("tool round trip:", "CALLED " + tool_name if tool_name else "NO TOOL CALL", "|", usage, "|", (text or "")[:80])

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    results = list(pool.map(one_call, range(N)))
elapsed = time.time() - t0
after = balance()
ok = sum(r[0] for r in results)
lat = sorted(r[1] for r in results if r[0])
errors = sorted({r[4] for r in results if not r[0]})
in_tok = sum(r[2] for r in results)
out_tok = sum(r[3] for r in results)
summary = {"model": MODEL, "n": N, "workers": WORKERS, "ok": ok,
           "median_latency_s": round(lat[len(lat) // 2], 2) if lat else None,
           "p90_latency_s": round(lat[int(len(lat) * 0.9)], 2) if lat else None,
           "elapsed_s": round(elapsed, 1), "input_tokens": in_tok, "output_tokens": out_tok,
           "tool_call_ok": bool(tool_name), "balance_before": before, "balance_after": after, "errors": errors}
print(json.dumps(summary, indent=1, default=str))
out = RESULTS_DIR / "smoke_tests.jsonl"
with out.open("a", encoding="utf-8") as f:
    f.write(json.dumps(summary, default=str) + "\n")
