"""
hybrid_st_flow_zic.py
---------------------
Hybrid Spatio-Temporal Graph Convolutional Flow Matching & Censored Operator (ST-GCN-CFM-ZIC).

Fuses:
1. Spatial Graph Convolution (GCN) on OD commute adjacency and geographical distance graphs.
2. Continuous Flow Matching (CFM) with Optimal Transport displacement interpolation:
   x_t = (1 - t) * x_0 + t * x_1,  u_t = x_1 - x_0
   along with a 10-step Euler ODE solver for smooth velocity field integration.
3. Zero-Inflated Censored (ZIC) Head for structural zero detection and latent demand recovery.
4. Physical Mass Conservation on dynamic bicycle returns.
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_normalized_graph(adj_matrix, k=15, add_self_loop=True):
    """
    Constructs a normalized, sparse-pruned graph convolution matrix A_norm.
    """
    A = adj_matrix.copy().astype(np.float32)
    N = len(A)
    np.fill_diagonal(A, 0.0)
    
    # Keep top-k strongest edges per row
    keep = np.argsort(-A, axis=1)[:, :k]
    A_sparse = np.zeros_like(A)
    rows = np.arange(N)[:, None]
    A_sparse[rows, keep] = np.log1p(A[rows, keep])
    
    if add_self_loop:
        A_sparse += np.eye(N, dtype=np.float32)
        
    deg = A_sparse.sum(axis=1, keepdims=True)
    A_norm = A_sparse / np.maximum(deg, 1e-7)
    return torch.from_numpy(A_norm).float()


class SinusoidalTimeEmbedding(nn.Module):
    """Continuous time t in [0, 1] positional embedding for Flow Matching."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim)
        args = t[:, None] * freqs[None, :] * 100.0
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class SpatialGraphConvBlock(nn.Module):
    """
    Spatial Graph Convolution block that diffuses features across the city graph:
    H' = GELU(A_od @ H @ W_od + A_dist @ H @ W_dist + H)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w_od = nn.Linear(in_features, out_features)
        self.w_dist = nn.Linear(in_features, out_features)
        self.w_skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        self.norm = nn.LayerNorm(out_features)

    def forward(self, h, A_od, A_dist):
        """
        h: [B, N, in_features]
        A_od: [N, N]
        A_dist: [N, N]
        """
        # Graph message passing
        msg_od = torch.matmul(A_od, h)      # [B, N, in_features]
        msg_dist = torch.matmul(A_dist, h)  # [B, N, in_features]

        out = self.w_od(msg_od) + self.w_dist(msg_dist) + self.w_skip(h)
        return self.norm(F.gelu(out))


class HybridFlowZICModel(nn.Module):
    """
    End-to-End Hybrid Architecture:
    ST-GCN Backbone + Continuous Flow Matching (CFM) + Zero-Inflated Censored Head.
    """
    def __init__(self, n_stations, od_matrix, dist_matrix=None, in_channels=2, hidden_dim=64):
        super().__init__()
        self.n_stations = n_stations
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Graph Laplacians
        A_od = build_normalized_graph(od_matrix, k=15)
        self.register_buffer('A_od', A_od)

        if dist_matrix is not None:
            # Invert distance: closer = stronger
            dist_sim = np.exp(-dist_matrix / 2.0)
            A_dist = build_normalized_graph(dist_sim, k=15)
        else:
            A_dist = A_od.clone()
        self.register_buffer('A_dist', A_dist)

        # 1. Spatio-Temporal GCN Backbone
        self.station_emb = nn.Embedding(n_stations, hidden_dim)
        self.hour_emb = nn.Embedding(24, 16)
        self.dow_emb = nn.Embedding(7, 8)
        self.month_emb = nn.Embedding(12, 8)

        self.input_fusion = nn.Linear(hidden_dim + 32, hidden_dim)
        self.gcn_layer1 = SpatialGraphConvBlock(hidden_dim, hidden_dim)
        self.gcn_layer2 = SpatialGraphConvBlock(hidden_dim, hidden_dim)

        # 2. Continuous Flow Matching Velocity Network (for flow states x_t)
        self.cfm_time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.flow_input_proj = nn.Linear(in_channels, hidden_dim)
        self.flow_gcn = SpatialGraphConvBlock(hidden_dim, hidden_dim)
        self.flow_velocity_out = nn.Linear(hidden_dim, in_channels)

        # 3. Structural Zero Gate Head (ZIC)
        self.zero_gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        nn.init.constant_(self.zero_gate_mlp[-1].bias, -2.0)

        # 4. Latent Demand Head
        self.macro_net = nn.Sequential(
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.spatial_potential_net = nn.Linear(hidden_dim, 1)

    def extract_gcn_representations(self, hour, weekday, month):
        """
        Computes GCN node representations conditioned on spatio-temporal context.
        Returns:
            H_gcn: [B, N, hidden_dim]
        """
        B, N = hour.shape[0], self.n_stations
        time_ctx = torch.cat([
            self.hour_emb(hour),
            self.dow_emb(weekday),
            self.month_emb(month)
        ], dim=-1) # [B, 32]
        
        ctx_expanded = time_ctx.unsqueeze(1).expand(B, N, -1) # [B, N, 32]
        st_emb = self.station_emb.weight.unsqueeze(0).expand(B, N, -1) # [B, N, hidden_dim]

        h0 = self.input_fusion(torch.cat([st_emb, ctx_expanded], dim=-1)) # [B, N, hidden_dim]

        h1 = self.gcn_layer1(h0, self.A_od, self.A_dist) # [B, N, hidden_dim]
        h2 = self.gcn_layer2(h1, self.A_od, self.A_dist) # [B, N, hidden_dim]
        return h2, time_ctx

    def compute_cfm_velocity(self, x_t, t_scalar, H_gcn):
        """
        Evaluates velocity field v_theta(x_t, t, H_gcn).
        Args:
            x_t: [B, N, in_channels]
            t_scalar: [B] in [0, 1]
            H_gcn: [B, N, hidden_dim]
        Returns:
            v_theta: [B, N, in_channels]
        """
        t_emb = self.cfm_time_mlp(t_scalar).unsqueeze(1) # [B, 1, hidden_dim]
        h_flow = self.flow_input_proj(x_t) + t_emb + H_gcn
        h_flow_gcn = self.flow_gcn(h_flow, self.A_od, self.A_dist)
        v_theta = self.flow_velocity_out(h_flow_gcn)
        return v_theta

    def forward_cfm_loss(self, x_1, x_0, H_gcn):
        """
        Conditional Flow Matching Loss along Optimal Transport displacement trajectory.
        x_t = (1 - t) * x_0 + t * x_1
        target velocity u_t = x_1 - x_0
        """
        B = x_1.shape[0]
        t = torch.rand(B, device=x_1.device)
        t_broad = t.view(B, 1, 1)

        x_t = (1.0 - t_broad) * x_0 + t_broad * x_1
        u_t = x_1 - x_0

        v_pred = self.compute_cfm_velocity(x_t, t, H_gcn)
        loss_cfm = F.mse_loss(v_pred, u_t)
        return loss_cfm

    @torch.no_grad()
    def sample_flow_ode(self, x_0, H_gcn, n_steps=10):
        """
        Integrates continuous velocity field from dawn state x_0 to peak state x_1
        using a 10-step Euler ODE solver.
        """
        x_t = x_0.clone()
        dt = 1.0 / n_steps
        B = x_t.shape[0]

        for step in range(n_steps):
            t_val = step * dt
            t_scalar = torch.full((B,), t_val, device=x_t.device, dtype=torch.float32)
            v = self.compute_cfm_velocity(x_t, t_scalar, H_gcn)
            x_t = x_t + dt * v

        return x_t # Generated peak state

    def forward(self, hour, weekday, month, station_mask=None):
        """
        Full forward pass producing:
        - H_gcn: [B, N, hidden_dim]
        - p_zero: [B, N] structural zero probability
        - lambda_rent: [B, N] latent demand intensity
        - mu_return: [B, N] mass-conserving return flux
        """
        B, N = hour.shape[0], self.n_stations
        H_gcn, time_ctx = self.extract_gcn_representations(hour, weekday, month)

        # 1. Structural Zero Gate
        logits_z = self.zero_gate_mlp(H_gcn).squeeze(-1) # [B, N]
        p_zero = torch.sigmoid(logits_z)
        if station_mask is not None:
            p_zero = torch.where(station_mask, p_zero, torch.ones_like(p_zero) * 0.999)

        # 2. Latent Demand (Macro M * Spatial Softmax pi)
        M = F.softplus(self.macro_net(time_ctx).squeeze(-1)) + 1.0 # [B]
        V_spatial = self.spatial_potential_net(H_gcn).squeeze(-1)  # [B, N]
        if station_mask is not None:
            V_spatial = V_spatial.masked_fill(~station_mask, -1e9)
        pi = F.softmax(V_spatial, dim=-1) # [B, N]
        lambda_rent = M.unsqueeze(1) * pi # [B, N]

        # Effective active demand
        lambda_eff = (1.0 - p_zero) * lambda_rent

        # 3. Mass-Conserving Physical Advection via OD Graph:
        # mu = A_od.T @ lambda_eff (Row conservation)
        # In batch: mu_return[b] = torch.matmul(lambda_eff[b], A_od)
        mu_return = torch.matmul(lambda_eff, self.A_od)

        return {
            'H_gcn': H_gcn,
            'p_zero': p_zero,
            'lambda_rent': lambda_rent,
            'lambda_effective': lambda_eff,
            'mu_return': mu_return,
            'M': M,
            'pi': pi
        }
