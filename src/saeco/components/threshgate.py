import torch
import torch.nn as nn


class ThreshGate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, acts, gate_acts: torch.Tensor):
        return acts.relu() * self.gate(gate_acts)

    def gate(self, gate_acts):
        return gate_acts > 0

    def soft_gate(self, gate_acts):
        return torch.sigmoid(gate_acts)


class SoftenedThreshGate(nn.Module):
    def __init__(self, threshgate: ThreshGate):
        super().__init__()
        self.threshgate = threshgate

    def forward(self, acts, gate_acts: torch.Tensor):
        return acts.relu() * self.threshgate.soft_gate(gate_acts)


class RandSoftThreshGate(ThreshGate):
    def __init__(self, p_soft=0.5):
        super().__init__()
        self.p_soft = p_soft

    def gate(self, gate_acts):
        if self.training:
            rand = torch.rand_like(gate_acts) < self.p_soft
            return torch.where(rand, super().gate(gate_acts), self.soft_gate(gate_acts))
        return super().gate(gate_acts)
