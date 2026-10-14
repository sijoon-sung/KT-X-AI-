import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class UltraResidualBlock(nn.Module):
    def __init__(self, dim=384, expand_dim=768, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.gcn = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, expand_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expand_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, adj, x_raw):
        h = self.norm1(x)
        g_out = torch.matmul(adj, self.gcn(h))
        x = x + F.gelu(g_out) + 0.1 * x_raw
        h2 = self.norm2(x)
        mlp_out = self.mlp(h2)
        out = x + mlp_out
        return out

class UltraSTGNN(nn.Module):
    def __init__(self, in_dim=20, hidden_dim=384, expand_dim=768, num_blocks=8):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.blocks = nn.ModuleList([
            UltraResidualBlock(hidden_dim, expand_dim) for _ in range(num_blocks)
        ])
        self.jk_proj = nn.Sequential(
            nn.Linear(hidden_dim * (num_blocks + 1), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.zero_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )
        self.rate_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )

    def forward(self, x, adj):
        h0 = self.in_proj(x)
        h = h0
        intermediates = [h0]
        for block in self.blocks:
            h = block(h, adj, h0)
            intermediates.append(h)
        jk = self.jk_proj(torch.cat(intermediates, dim=-1))
        pi = torch.sigmoid(self.zero_head(jk))
        lam = F.softplus(self.rate_head(jk)) + 1e-4
        return pi, lam

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Active device:", dev)

model = UltraSTGNN(in_dim=20, hidden_dim=384, expand_dim=768, num_blocks=8).to(dev)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {n_params:,}")

B, N = 32, 1283
x = torch.randn(B, N, 20, device=dev)
adj = torch.randn(N, N, device=dev)

t0 = time.time()
pi, lam = model(x, adj)
loss = (pi.sum() + lam.sum())
loss.backward()
torch.cuda.synchronize()
print(f"Full Forward+Backward time: {time.time()-t0:.4f}s")
print(f"Allocated VRAM: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / (1024**2):.1f} MB")
