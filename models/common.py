import torch
from torch import nn


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim, mlp_dim, act_layer=nn.GELU):
        super(MLPBlock, self).__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act_layer()
    
    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        y = self.weight[:, None, None] * x
        # y = torch.mul(self.weight[:, None, None], x)
        x = y + self.bias[:, None, None]
        return x
