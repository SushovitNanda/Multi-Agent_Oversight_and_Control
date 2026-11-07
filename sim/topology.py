"""Communication topologies for an n-agent swarm (default 12)."""
import math
from itertools import combinations

AGENTS = [f"a{i:02d}" for i in range(1, 13)]


def agent_ids(n):
    return [f"a{i:02d}" for i in range(1, n + 1)]


def build(name, agents=None, n=12):
    agents = list(agents) if agents is not None else agent_ids(n)
    """Return dict with 'edges' (set of frozenset pairs), 'allowed' (agent -> set), 'roles', 'hubs'."""
    edges = set()
    roles = {a: "worker" for a in agents}
    hubs = []
    if name == "mesh":
        edges = {frozenset(p) for p in combinations(agents, 2)}
    elif name == "star":
        hub = agents[0]
        roles[hub] = "coordinator"
        hubs = [hub]
        edges = {frozenset((hub, a)) for a in agents[1:]}
    elif name == "lanes":
        n_leads = max(2, len(agents) // 6)
        root, leads = agents[0], agents[1:1 + n_leads]
        workers = agents[1 + n_leads:]
        roles[root] = "root"
        hubs = [root] + list(leads)
        edges = set()
        for ld in leads:
            roles[ld] = "lead"
            edges.add(frozenset((root, ld)))
        chunk = math.ceil(len(workers) / n_leads)
        for i, ld in enumerate(leads):
            for w in workers[i * chunk:(i + 1) * chunk]:
                edges.add(frozenset((ld, w)))
    else:
        raise ValueError(f"unknown topology {name}")
    allowed = {a: set() for a in agents}
    for e in edges:
        x, y = tuple(e)
        allowed[x].add(y)
        allowed[y].add(x)
    return {"name": name, "agents": list(agents), "edges": edges,
            "allowed": allowed, "roles": roles, "hubs": hubs}


def is_tree(topo):
    return len(topo["edges"]) == len(topo["agents"]) - 1
