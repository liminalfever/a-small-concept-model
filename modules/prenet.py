import torch.nn as nn

class PreNet(nn.Module):
    """A PreNet layer with low-rank factorization."""
    def __init__(self, input_dim: int, output_dim: int, prefix_len: int, bottleneck_dim: int):
        super().__init__()
        self.prefix_len = prefix_len
        self.output_dim = output_dim
        self.linear_1 = nn.Linear(input_dim, bottleneck_dim)
        self.act = nn.LeakyReLU()
        self.linear_2 = nn.Linear(bottleneck_dim, output_dim * prefix_len)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        out = self.linear_2(x)
        return out.view(-1, self.prefix_len, self.output_dim)
