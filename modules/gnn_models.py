import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

pdf_path="docs\medbook1\medical_book.pdf"

class NodeAlignerGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.gnn1 = GATConv(in_dim, hidden_dim, heads=2, concat=True)
        self.gnn2 = GATConv(hidden_dim * 2, out_dim, heads=1)
        self.query_mlp = torch.nn.Sequential(
            torch.nn.Linear(out_dim + 384, out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(out_dim, 1)
        )

    def forward(self, x, edge_index, batch, query_emb):
        x = self.gnn1(x, edge_index)
        x = self.gnn2(x, edge_index)
        q = query_emb.unsqueeze(0).repeat(x.size(0), 1)
        scores = self.query_mlp(torch.cat([x, q], dim=1)).squeeze()
        return x, F.softmax(scores, dim=0)