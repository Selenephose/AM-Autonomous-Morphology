"""
AM-Autonomous-Morphology
Layer 1 — Brainstem Kernel

Minimal cognitive organism that:
• compresses sensory state into latent space
• learns predictive world dynamics
• discovers reusable skills via abstraction
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random

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
# Skill Memory (Reusable Abstractions)
# ================================

class SkillMemory:
    def __init__(self):
        self.skills = []

    def store(self, z):
        self.skills.append(z.detach().clone())

    def sample(self):
        if not self.skills:
            return None
        return random.choice(self.skills)


# ================================
# Core Organism
# ================================

class Brainstem:
    def __init__(self, state_dim, action_dim):
        self.encoder = AbstractionEncoder(state_dim)
        self.world_model = WorldModel(action_dim)
        self.memory = SkillMemory()

        self.optim = optim.Adam(
            list(self.encoder.parameters()) + list(self.world_model.parameters()),
            lr=3e-4
        )

    def step(self, state, action, next_state):
        z = self.encoder(state)
        z_next = self.encoder(next_state)
        z_pred = self.world_model(z, action)

        loss = ((z_pred - z_next) ** 2).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if loss.item() > 0.1:
            self.memory.store(z)

        return loss.item()

