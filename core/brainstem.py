"""
AM-Autonomous-Morphology
Layer 1 — Brainstem Kernel

Autonomous cognitive organism that:
• compresses perception into latent abstractions
• learns predictive world dynamics
• discovers and tracks symbolic concepts
• promotes reusable skills automatically
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import math

LATENT_DIM = 8


# ================================
# Sensory Abstraction
# ================================

class AbstractionEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, LATENT_DIM)
        )

    def forward(self, x):
        return self.net(x)


# ================================
# World Model (Predictive Dynamics)
# ================================

class WorldModel(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, LATENT_DIM)
        )

    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))


# ================================
# Symbol Memory (Concept Lifecycle)
# ================================

class SymbolMemory:
    def __init__(self):
        self.usages = {}
        self.transitions = {}
        self.tasks = {}
        self.reusable = {}

    def observe(self, symbol, task, next_symbol):
        self.usages[symbol] = self.usages.get(symbol, 0) + 1
        self.tasks.setdefault(symbol, set()).add(task)
        if next_symbol:
            self.transitions.setdefault(symbol, set()).add(next_symbol)

        # Promotion rule
        if self.usages[symbol] > 12 and len(self.tasks[symbol]) > 1:
            self.reusable[symbol] = True

    def summary(self):
        return {
            s: {
                "usage": self.usages[s],
                "tasks": list(self.tasks.get(s, [])),
                "reusable": self.reusable.get(s, False),
                "transitions": list(self.transitions.get(s, []))
            }
            for s in self.usages
        }


# ================================
# Latent Symbolizer
# ================================

class LatentSymbolizer:
    def __init__(self, k=12):
        self.k = k
        self.centroids = []
        self.ids = []

    def rebuild(self, latents):
        if len(latents) < self.k:
            return
        self.centroids = []
        self.ids = []
        for i in range(self.k):
            idx = int((i / self.k) * len(latents))
            self.centroids.append(latents[idx])
            self.ids.append(f"S{i}")

    def assign(self, z):
        if not self.centroids:
            return "S0"
        dists = [torch.norm(z - c).item() for c in self.centroids]
        return self.ids[dists.index(min(dists))]


# ================================
# Core Organism
# ================================

class Brainstem:
    def __init__(self, state_dim, action_dim):
        self.encoder = AbstractionEncoder(state_dim)
        self.world_model = WorldModel(action_dim)

        self.symbolizer = LatentSymbolizer()
        self.symbol_memory = SymbolMemory()
        self.last_symbol = None
        self.latent_history = []

        self.optim = optim.Adam(
            list(self.encoder.parameters()) + list(self.world_model.parameters()),
            lr=3e-4
        )

    def step(self, state, action, next_state, task="task0"):
        z = self.encoder(state)
        z_next = self.encoder(next_state)
        z_pred = self.world_model(z, action)

        loss = ((z_pred - z_next) ** 2).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # store latent history
        self.latent_history.append(z.detach())

        # assign symbolic concept
        symbol = self.symbolizer.assign(z)

        # update symbolic memory
        self.symbol_memory.observe(symbol, task, self.last_symbol)
        self.last_symbol = symbol

        return loss.item(), symbol

    def end_episode(self):
        # rebuild concept centroids
        self.symbolizer.rebuild(self.latent_history)
        self.latent_history.clear()

        print("\n=== SYMBOL MAP ===")
        for s, data in self.symbol_memory.summary().items():
            if data["reusable"]:
                print("REUSABLE SKILL:", s, data)
