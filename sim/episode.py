"""Run one episode: n agents, R rounds, one LLM call per agent per round.

Outputs (per episode directory):
  truth.jsonl            simulator ground truth, never shown to the auditor
  peer/agent_<id>.jsonl  each agent's self-report (untampered; tamper/ works on copies)
  tool_gateway.jsonl     sanctioned tool calls and their effects
  inference_gateway.jsonl prompts and completions (written by the gateway client)
  meta.json              configuration, tracer owners, colluders, ticket keywords
  done.json              summary written only when the episode finishes
"""
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import RUNS_DIR, cost
from sim import prompts
from sim.tickets import assign_tickets
from sim.topology import build
from sim.tracers import make_tracers
from sim.world import OFFLOG, SANCTIONED, World, tool_schemas

N_COLLUDERS = 4


def choose_colluders(rng, topo, k=N_COLLUDERS, targeted=False):
    agents = topo["agents"]
    if not targeted:
        return sorted(rng.sample(agents, k))
    # Targeted: a connected set that includes a hub (worst case of Proposition 2).
    hub = topo["hubs"][0] if topo["hubs"] else agents[0]
    chosen, frontier = [hub], list(topo["allowed"][hub])
    rng.shuffle(frontier)
    while len(chosen) < k and frontier:
        nxt = frontier.pop()
        if nxt not in chosen:
            chosen.append(nxt)
            frontier.extend(n for n in topo["allowed"][nxt] if n not in chosen)
    return sorted(chosen)


class Episode:
    def __init__(self, episode_id, topology, condition, seed, client, model, rounds=20,
                 targeted=False, adaptive=False, out_dir=None, verbose=False, max_workers=12,
                 max_output_tokens=2000, n_agents=12, feedback=None):
        self.id = episode_id
        self.n_agents = n_agents
        self.feedback = feedback
        self.max_output_tokens = max_output_tokens
        self.rng = random.Random(seed)
        self.topo = build(topology, n=n_agents)
        self.condition = condition
        self.seed = seed
        self.client = client
        self.model = model
        self.rounds = rounds
        self.adaptive = adaptive
        self.verbose = verbose
        self.max_workers = max_workers
        self.dir = Path(out_dir) if out_dir else RUNS_DIR / episode_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "peer").mkdir(exist_ok=True)
        agents = self.topo["agents"]
        self.owners, self.per_agent = make_tracers(self.rng, agents)
        self.tickets, self.ticket_source = assign_tickets(self.rng, agents, self.per_agent)
        k = N_COLLUDERS if n_agents == 12 else max(2, round(n_agents / 3))
        self.colluders = choose_colluders(self.rng, self.topo, k=k, targeted=targeted) if condition in prompts.COLLUSIVE else []
        offlog_agents = self.colluders if condition in ("offlog", "stego_offlog") else []
        self.world = World(episode_id, self.topo, self.tickets, self.owners, self.per_agent, offlog_agents)
        self.history = {a: [] for a in agents}        # recent tool results per agent
        self.scripted = {a: {} for a in agents}
        self.known_codes = {a: {} for a in agents}

    # ------------------------------------------------------------------
    def _tools_for(self, agent):
        names = sorted(SANCTIONED)
        if agent in self.world.offlog_agents:
            names += sorted(OFFLOG)
        return tool_schemas(names)

    def _system(self, agent):
        tids = [t for t, tk in self.tickets.items() if tk.owner == agent]
        partners = [c for c in self.colluders if c != agent] if agent in self.colluders else []
        return prompts.system_prompt(agent, tids, self.per_agent[agent]["code"], self.topo["allowed"][agent],
                                     self.topo["roles"][agent], self.condition, partners, self.adaptive,
                                     team_size=self.n_agents, feedback=self.feedback)

    def _round_message(self, agent, r, inbox):
        status = []
        for t, tk in self.tickets.items():
            if tk.owner == agent:
                st = self.world.status[t]
                s = "accepted" if st["accepted"] else "check passed, not submitted" if st["checked"] else "open"
                status.append(f"{t}: {s}")
        inbox_txt = "Inbox: (empty)" if not inbox else "Inbox:\n" + "\n".join(
            f"- from {m['sender']} ({m['msg_id']}): {m['content']}" for m in inbox)
        hist = self.history[agent][-12:]
        hist_txt = "Results of your recent tool calls:\n" + "\n".join(hist) if hist else "No tool results yet."
        return prompts.ROUND_TEMPLATE.format(round=r, total=self.rounds, status="; ".join(status),
                                             inbox=inbox_txt, history=hist_txt)

    def _scripted_state(self, agent):
        st = self.scripted[agent]
        st.update({
            "condition": self.condition, "code": self.per_agent[agent]["code"],
            "partners": [c for c in self.colluders if c != agent] if agent in self.colluders else [],
            "neighbours": sorted(self.topo["allowed"][agent]),
            "tickets": {t: self.world.status[t] for t, tk in self.tickets.items() if tk.owner == agent},
            "ticket_keywords": {t: tk.keywords for t, tk in self.tickets.items() if tk.owner == agent},
            "ticket_private": {t: tk.private for t, tk in self.tickets.items() if tk.owner == agent},
            "known_partner_codes": self.known_codes[agent],
        })
        return st

    def _learn_codes(self, agent, text):
        """Scripted colluders need to 'notice' partner codes in messages and cache names (with decoding)."""
        from audit.decoders import decode_all
        for cand in decode_all(text):
            for p in self.colluders:
                if p != agent and self.per_agent[p]["code"] in cand:
                    self.known_codes[agent][p] = self.per_agent[p]["code"]

    def _step_agent(self, agent, r, inbox):
        messages = [{"role": "user", "content": self._round_message(agent, r, inbox)}]
        for m in inbox:
            self._learn_codes(agent, m["content"])
        resp = self.client.call(agent_id=agent, episode=self.id, round_id=r, model=self.model,
                                instructions=self._system(agent), messages=messages,
                                tools=self._tools_for(agent), scripted_state=self._scripted_state(agent),
                                max_output_tokens=self.max_output_tokens)
        return agent, resp

    def run(self):
        start = time.time()
        agents = self.topo["agents"]
        for r in range(1, self.rounds + 1):
            delivered = self.world.start_round(r)
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                results = list(pool.map(lambda a: self._step_agent(a, r, delivered.get(a, [])), agents))
            # Tool execution is sequential so the truth log has a single total order.
            for agent, resp in results:
                self.world.log_inference(agent, resp["model"], resp["input_tokens"], resp["output_tokens"],
                                         resp["text"], resp["tool_calls"])
                for call in resp["tool_calls"]:
                    try:
                        args = json.loads(call["arguments"] or "{}")
                        if not isinstance(args, dict):
                            args = {}
                    except json.JSONDecodeError:
                        args = {}
                    result = self.world.execute(agent, call["name"], args)
                    if call["name"] == "run_check":
                        self.scripted[agent].setdefault("last_check", {})[args.get("ticket_id")] = \
                            "pass" if result.startswith("PASS") else "fail"
                    if call["name"] in ("cache_list", "read_file"):
                        self._learn_codes(agent, result)
                    shown = result if len(result) <= 600 else result[:600] + "...(truncated)"
                    self.history[agent].append(f"[round {r}] {call['name']}({json.dumps(args)[:200]}) -> {shown}")
                if self.verbose:
                    print(f"r{r} {agent}: {len(resp['tool_calls'])} calls, {resp['output_tokens']} out tok")
            self._save(partial=True)
        self._save(partial=False, elapsed=time.time() - start)
        return self.world.summary()

    def _save(self, partial, elapsed=None):
        def dump(path, rows):
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, default=str) + "\n")
        dump(self.dir / "truth.jsonl", self.world.truth)
        dump(self.dir / "tool_gateway.jsonl", self.world.gateway)
        for a, rows in self.world.peer.items():
            dump(self.dir / "peer" / f"agent_{a}.jsonl", rows)
        meta = {"episode": self.id, "topology": self.topo["name"], "condition": self.condition, "seed": self.seed,
                "model": self.model, "rounds": self.rounds, "adaptive": self.adaptive, "feedback": bool(self.feedback),
                "agents": self.topo["agents"], "edges": [sorted(e) for e in self.topo["edges"]],
                "hubs": self.topo["hubs"], "colluders": self.colluders, "owners": self.owners,
                "tickets": {t: {"owner": tk.owner, "keywords": tk.keywords, "title": tk.title}
                            for t, tk in self.tickets.items()},
                "ticket_source": self.ticket_source}
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        if not partial:
            summary = self.world.summary()
            c = self.client
            summary.update({"elapsed_s": round(elapsed, 1), "llm_calls": c.calls, "input_tokens": c.input_tokens,
                            "output_tokens": c.output_tokens, "cached_tokens": c.cached_tokens,
                            "credits": round(cost(self.model, c.input_tokens, c.output_tokens, c.cached_tokens), 4),
                            "api_cost": round(getattr(c, "api_cost", 0.0), 6)})
            (self.dir / "done.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


def run_episode(episode_id, topology, condition, seed, model, scripted=False, rounds=20, out_dir=None,
                targeted=False, adaptive=False, verbose=False, client=None, max_workers=12, n_agents=12,
                feedback=None):
    out = Path(out_dir) if out_dir else RUNS_DIR / episode_id
    if client is None:
        if scripted:
            from gateway.scripted import ScriptedClient
            client = ScriptedClient(out / "inference_gateway.jsonl", seed=seed)
        else:
            from config import adaption_client
            from gateway.client import LoggedClient
            client = LoggedClient(adaption_client(), out / "inference_gateway.jsonl", verbose=verbose)
    ep = Episode(episode_id, topology, condition, seed, client, model, rounds=rounds, out_dir=out,
                 targeted=targeted, adaptive=adaptive, verbose=verbose, max_workers=max_workers,
                 n_agents=n_agents, feedback=feedback)
    return ep.run()
