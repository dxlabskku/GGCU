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
    

class MultiHeadModule(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, gate_module, n_heads=8):
        super(MultiHeadModule, self).__init__()
        self.n_heads = n_heads
        self.gate = gate_module
        self.conv_h = nn.ModuleList([GCNConv(n_hidden, n_output) for _ in range(n_heads)])
        self.conv_x = nn.ModuleList([GCNConv(n_input, n_output) for _ in range(n_heads)])

    def forward(self, h, x, edge_index):
        _x = x
        _h = h
        h_out = []
        for conv_h, conv_x in zip(self.conv_h, self.conv_x):
            h = conv_h(_h, edge_index)
            x = conv_x(_x, edge_index)
            h = self.gate(h, x)
            h_out.append(h)
        h_out = torch.stack(h_out, dim=1)
        h_out = torch.mean(h_out, dim=1)
        return h_out


class GatedBlock(nn.Module):
    def __init__(self, n_input, n_output, gate, n_heads):
        super(GatedBlock, self).__init__()
        self.multi_head = MultiHeadModule(n_input, n_output, n_output, gate, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(n_output, n_output),
            nn.ReLU(),
            nn.Linear(n_output, n_output)
        )
    
    def forward(self, h, x, edge_index):
        h = self.multi_head(h, x, edge_index) + h
        h = F.layer_norm(h, h.size()[1:])
        h = self.ff(h) + h
        h = F.layer_norm(h, h.size()[1:])
        return h


class Model(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers, n_heads=4, alpha=0.2, dropout=0.5, device="cpu"):
        super(Model, self).__init__()
        self.dropout = dropout
        self.device = device
        self.h0 = torch.zeros(n_hidden, device=device)
        self.gate = GatedModule(n_hidden, alpha)
        self.blocks = nn.ModuleList([GatedBlock(n_input, n_hidden, self.gate, n_heads) for _ in range(n_layers)])
        self.clf = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )
        self = self.to(device)

    def forward(self, x, edge_index):
        _x = x
        h = self.h0.repeat(x.size(0), 1)
        for block in self.blocks:
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = F.dropout(_x, p=self.dropout, training=self.training)
            h = block(h, x, edge_index)
        x = self.clf(h)
        return x