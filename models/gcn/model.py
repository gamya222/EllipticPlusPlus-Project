"""A dependency-light two-layer GCN implemented with PyTorch sparse tensors."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GCN(nn.Module):
    """Two graph-convolution layers using symmetric normalized adjacency."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output_layer = nn.Linear(hidden_dim, 2, bias=True)
        self.dropout = dropout

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = torch.sparse.mm(adjacency, features)
        hidden = F.relu(self.input_layer(hidden))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = torch.sparse.mm(adjacency, hidden)
        return self.output_layer(hidden)
