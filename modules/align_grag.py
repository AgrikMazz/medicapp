"""Small, dependency-light implementation of Align-GRAG's dual aligner.

The paper uses a GNN node scorer, anchor-derived KL supervision, and a
graph/rationale contrastive objective.  This module uses a two-layer
GraphSAGE encoder implemented with ordinary PyTorch so the experiment does not
depend on torch_geometric.  Candidate subgraphs and supervision construction
are handled by the benchmark runner.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MeanSAGELayer(nn.Module):
    """GraphSAGE mean-neighbor layer over a row-normalized adjacency matrix."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.self_projection = nn.Linear(input_dim, output_dim)
        self.neighbor_projection = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbors = adjacency @ x
        return self.self_projection(x) + self.neighbor_projection(neighbors)


class AlignGRAGAligner(nn.Module):
    """Query-conditioned node alignment plus a graph/text projection head."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        self.gnn1 = MeanSAGELayer(input_dim, hidden_dim)
        self.gnn2 = MeanSAGELayer(hidden_dim, input_dim)
        self.node_scorer = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
        )
        # Shared projection follows the paper's graph/rationale alignment head.
        self.shared_projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
        )
        self.dropout = 0.1

    def encode_graph(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        x = F.dropout(node_features, p=self.dropout, training=self.training)
        x = F.relu(self.gnn1(x, adjacency))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.relu(self.gnn2(x, adjacency))

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_embeddings = self.encode_graph(node_features, adjacency)
        query_rows = query_embedding.unsqueeze(0).expand(node_embeddings.size(0), -1)
        node_logits = self.node_scorer(
            torch.cat((node_embeddings, query_rows), dim=-1)
        ).squeeze(-1)
        graph_embedding = node_embeddings.mean(dim=0)
        return node_embeddings, node_logits, graph_embedding

    def project_graph_text(
        self, graph_embedding: torch.Tensor, text_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        graph_projected = F.normalize(self.shared_projection(graph_embedding), dim=-1)
        text_projected = F.normalize(self.shared_projection(text_embedding), dim=-1)
        return graph_projected, text_projected


def symmetric_info_nce(
    graph_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Bidirectional in-batch graph-to-rationale contrastive loss."""
    if graph_embeddings.size(0) < 2:
        return graph_embeddings.new_zeros(())
    graph_embeddings = F.normalize(graph_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    logits = graph_embeddings @ text_embeddings.transpose(0, 1) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def anchor_kl_loss(node_logits: torch.Tensor, anchor_distribution: torch.Tensor) -> torch.Tensor:
    """KL(p_anchor || p_predicted), as in Align-GRAG's node-level objective."""
    target = anchor_distribution.clamp_min(1e-8)
    target = target / target.sum()
    return F.kl_div(F.log_softmax(node_logits, dim=0), target, reduction="sum")
