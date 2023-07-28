
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv


class GatedModule(nn.Module):
    def __init__(self, n_latent, alpha=0.1):
        super(GatedModule, self).__init__()
        self.gate_f = nn.Linear(n_latent * 2, 1)
        self.gate_u = nn.Linear(n_latent * 2, 1)
        self.alpha = alpha

    def forward(self, h, x):
        gate = torch.cat([h, x], dim=1)
        f = self.alpha * torch.tanh(self.gate_f(gate))
        u = self.alpha * torch.tanh(self.gate_u(gate))
        h = (1 + f) * h + (1 + u) * x
        return h


class Model(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers, alpha=0.2, dropout=0.5, device="cpu"):
        super(Model, self).__init__()
        self.h0 = torch.zeros(n_hidden, device=device)
        self.conv_h = nn.ModuleList([GCNConv(n_hidden, n_hidden) for _ in range(n_layers)])
        self.conv_x = nn.ModuleList([GCNConv(n_input, n_hidden) for _ in range(n_layers)])
        self.res = nn.ModuleList([GCNConv(n_hidden, n_hidden) for _ in range(n_layers)])
        for conv in self.res:
            conv.lin.weight = nn.Parameter(torch.eye(n_hidden), requires_grad=False)
            conv.lin.bias = nn.Parameter(torch.zeros(n_hidden), requires_grad=False)
        self.gate = GatedModule(n_hidden, alpha)
        self.fc = nn.Linear(n_hidden, n_output)
        self.dropout = dropout
        self.device = device

        self = self.to(device)

    def forward(self, x, edge_index):
        _x = x
        h = self.h0.repeat(x.size(0), 1)
        for conv_h, conv_x, res in zip(self.conv_h, self.conv_x, self.res):
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = F.dropout(_x, p=self.dropout, training=self.training)
            _h = conv_h(h, edge_index)
            x = conv_x(x, edge_index)
            h = self.gate(_h, x) + res(h, edge_index)
        x = self.fc(h)
        return x