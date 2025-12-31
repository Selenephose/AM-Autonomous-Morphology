import torch
import random
from math import sqrt

class MentalRolloutEngine:
    def __init__(self, world_model, action_dim, horizon=5):
        self.world_model = world_model
        self.action_dim = action_dim
        self.horizon = horizon

    def imagine(self, z):
        score = 0
        curr = z.clone()
        for _ in range(self.horizon):
            a = torch.randn(self.action_dim)
            next_z = self.world_model(curr, a)
            stability = torch.std(next_z).item()
            energy = torch.norm(next_z).item()
            score += energy - 0.3 * stability
            curr = next_z
        return score


class Cortex:
    def __init__(self, brainstem):
        self.brainstem = brainstem
        self.rollout = MentalRolloutEngine(
            brainstem.world_model,
            action_dim=brainstem.world_model.net[0].in_features - 8
        )
        self.meta_skills = []

    def evaluate_skill(self, z):
        score = self.rollout.imagine(z)
        if score > 1.0:
            self.meta_skills.append(z.detach().clone())
        return score
