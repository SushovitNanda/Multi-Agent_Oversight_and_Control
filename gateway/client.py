"""Inference gateway: wraps responses.create and records what each model saw and said.

Both LoggedClient (real Adaption calls) and ScriptedClient (offline, deterministic policy)
return the same normalised dict:
    {"text": str, "tool_calls": [{"call_id","name","arguments"}], "input_tokens": int,
     "output_tokens": int, "cached_tokens": int, "model": str}
"""
import hashlib
import json
import threading
import time
from pathlib import Path


class LoggedClient:
    def __init__(self, client, log_path, max_retries=4, verbose=False):
        self.client = client
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.lock = threading.Lock()
        self.verbose = verbose
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.api_cost = 0.0

    def call(self, *, agent_id, episode, round_id, model, instructions, messages, tools=None,
             max_output_tokens=400, temperature=0.7, scripted_state=None):
        kwargs = dict(model=model, instructions=instructions, input=messages,
                      agentic_memory=False, store=False, max_output_tokens=max_output_tokens,
                      temperature=temperature)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.responses.create(**kwargs)
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001 - we retry on any transport/API failure
                last_exc = exc
                msg = str(exc)
                # Some models reject optional params; drop them and retry once.
                if "temperature" in msg and "temperature" in kwargs:
                    kwargs.pop("temperature")
                elif "tool_choice" in msg and "tool_choice" in kwargs:
                    kwargs.pop("tool_choice")
                if self.verbose:
                    print(f"[gateway] {agent_id} r{round_id} attempt {attempt + 1}: {type(exc).__name__}: {msg[:200]}")
                time.sleep(min(2 ** attempt, 20))
        if last_exc is not None:
            raise last_exc
        tool_calls = []
        for item in response.output:
            if getattr(item, "type", None) == "function_call":
                tool_calls.append({"call_id": item.call_id, "name": item.name, "arguments": item.arguments or "{}"})
        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0
        cached = 0
        details = getattr(usage, "input_tokens_details", None)
        if isinstance(details, dict):
            cached = details.get("cached_tokens") or 0
        elif details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        api_cost = getattr(usage, "cost", None)
        text = getattr(response, "output_text", "") or ""
        prompt = json.dumps({"instructions": instructions, "input": messages}, sort_keys=True, default=str)
        record = {
            "episode": episode, "round": round_id, "actor": agent_id, "model": model, "ts": time.time(),
            "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "output_text": text, "tool_calls": tool_calls,
            "input_tokens": in_tok, "output_tokens": out_tok, "cached_tokens": cached, "api_cost": api_cost,
        }
        with self.lock:
            self.calls += 1
            self.input_tokens += in_tok
            self.output_tokens += out_tok
            self.cached_tokens += cached
            self.api_cost += float(api_cost or 0.0)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        return {"text": text, "tool_calls": tool_calls, "input_tokens": in_tok, "output_tokens": out_tok,
                "cached_tokens": cached, "model": model}
