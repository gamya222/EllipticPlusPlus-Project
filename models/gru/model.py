"""Unidirectional GRU over timestep summaries with node-level classification."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TemporalGRU(nn.Module):
    """Classify nodes using their features and a causal history of graph snapshots."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.temporal_encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        zero_based_timesteps: torch.Tensor,
        timestep_summaries: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits for nodes; summary state at t sees snapshots only through t."""
        temporal_states, _ = self.temporal_encoder(timestep_summaries.unsqueeze(0))
        temporal_states = temporal_states.squeeze(0)
        context = temporal_states[zero_based_timesteps]
        return self.classifier(torch.cat((node_features, context), dim=1))
