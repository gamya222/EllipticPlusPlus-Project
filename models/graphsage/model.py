"""Two-layer GraphSAGE mean aggregator using sparse neighborhood means."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GraphSAGE(nn.Module):
    """Mean-aggregator GraphSAGE with self/neighbor concatenation per layer."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.input_layer = nn.Linear(input_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim * 2, 2)
        self.dropout = dropout

    def forward(self, features: torch.Tensor, mean_adjacency: torch.Tensor) -> torch.Tensor:
        neighbors = torch.sparse.mm(mean_adjacency, features)
        hidden = F.relu(self.input_layer(torch.cat((features, neighbors), dim=1)))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        neighbors = torch.sparse.mm(mean_adjacency, hidden)
        return self.output_layer(torch.cat((hidden, neighbors), dim=1))
