import torch
import numpy as np
import math as pymath
import random

class SymbolPhaseMetrics:
    def __init__(self):
        self.usage = {}
        self.transitions = {}
        self.tasks = {}
        self.birth = {}
        self.death = {}
        self.reusable = {}
        self.general = {}
        self.epoch = 0

    def observe(self, sym, task, nxt):
        self.usage[sym] = self.usage.get(sym, 0) + 1
        self.tasks.setdefault(sym, set()).add(task)
        self.transitions.setdefault(sym, set()).add(nxt)
        self.birth.setdefault(sym, self.epoch)
        if self.usage[sym] > 15 and len(self.transitions[sym]) > 3:
            self.general[sym] = True
        if self.usage[sym] > 8 and len(self.tasks[sym]) > 2:
            self.reusable[sym] = True

    def collapse(self, sym):
        self.death[sym] = self.epoch

    def snapshot(self):
        return [{
            "symbol": s,
            "usage": self.usage[s],
            "general": self.general.get(s, False),
            "reusable": self.reusable.get(s, False),
            "tasks": list(self.tasks.get(s, [])),
            "born": self.birth.get(s),
            "dead": self.death.get(s)
        } for s in self.usage]

class HierarchicalSymbolGraph:
    def __init__(self):
        self.nodes = {}
        self.parents = {}
        self.programs = {}

    def add_symbol(self, sid, vec):
        self.nodes[sid] = np.array(vec)

    def add_edge(self, parent, child):
        self.parents[child] = parent

    def compose(self, mid, seq):
        self.programs[mid] = seq

    def execute(self, mid, z):
        for sid in self.programs.get(mid, []):
            z = self.nodes[sid]
        return z

class CounterfactualPlanner:
    def __init__(self, kernel):
        self.kernel = kernel
        self.rollout_width = 6
        self.rollout_depth = 6

    def _score_latent(self, z):
        entropy = -float(torch.sum(z * torch.log(z + 1e-8)))
        variance = float(torch.var(z))
        compression = -float(torch.mean(torch.abs(z)))
        return 0.6 * entropy + 0.3 * variance + 0.1 * compression

    def imagine(self, start_latent):
        best = None
        best_score = -1e9
        z0 = torch.tensor(start_latent, dtype=torch.float32)
        for _ in range(self.rollout_width):
            z = z0.clone()
            score = 0
            plan = []
            for _ in range(self.rollout_depth):
                a = torch.rand(self.kernel.action_dim) * 2 - 1
                z_next = self.kernel.wm(z.unsqueeze(0), a.unsqueeze(0)).squeeze(0)
                score += self._score_latent(z_next)
                plan.append(a.numpy().tolist())
                z = z_next.detach()
            if score > best_score:
                best_score = score
                best = plan
        return best

class AM_Cortex:
    def __init__(self, kernel, llm=None):
        self.kernel = kernel
        self.phase = SymbolPhaseMetrics()
        self.graph = HierarchicalSymbolGraph()
        self.planner = CounterfactualPlanner(kernel)
        self.llm = llm

    def observe(self, sym, task, nxt, vec):
        self.phase.observe(sym, task, nxt)
        self.graph.add_symbol(sym, vec)

    def plan(self, latent):
        return self.planner.imagine(latent)

    def metrics(self):
        return self.phase.snapshot()

    def compose_meta(self, symbols):
        meta_id = "M" + ''.join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
        self.graph.compose(meta_id, symbols)
        return meta_id

    def hierarchy(self):
        return {k: {"parent": self.graph.parents.get(k), "vec": self.graph.nodes[k].tolist()} for k in self.graph.nodes}
