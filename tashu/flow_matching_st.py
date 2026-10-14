"""Scaled Graph-Informed Spatio-Temporal Continuous Flow Matching (ST-CFM).

Theoretical Formulation:
  1. Flow Matching for Generative Modeling (Lipman et al., ICLR 2023; Zhang et al., ICML 2026):
     - OT Displacement Interpolation:
         x_t = (1 - t) * x_0 + t * x_1    (t in [0, 1])
         Target velocity field: u_t = x_1 - x_0
     - Prior distribution: x_0 ~ N(0, sigma^2 * I)
     - Target data: x_1 = log1p(y) in R^{B, N, 2} (Joint Rentals & Returns for all N stations)
  2. Scaled High-Capacity Architecture:
     - SpatioTemporalContextEncoder: Deep Residual GCN with LayerNorm, FFN expansion,
       and Jumping Knowledge (JK) multi-scale skip connections.
     - ScaledGraphVelocityFieldNet: High-capacity velocity field network with per-block
       temporal modulation, multi-layer GCN propagation, and jumping knowledge.
  3. Numerical Solvers:
     - Euler (1st order, fast 5-10 steps)
     - Midpoint (2nd order Runge-Kutta)
     - RK4 (4th order Classical Runge-Kutta)
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F

from scale_up_massive_model import load_enriched_dataset, prepare_deep_batch


# =========================================================================
# 1. Core Modules: Time Embedding & Deep Residual GCN Blocks
# =========================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal frequency embedding for continuous time t in [0, 1]."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B]
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim)
        args = t[:, None] * freqs[None, :] * 100.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class UltraResidualGCNBlock(nn.Module):
    """
    Deep Residual Graph Convolution Block with LayerNorm, Raw Input Skip,
    and Feed-Forward Network (FFN) with 2x expansion to prevent over-smoothing.
    """
    def __init__(self, dim, expand_dim=None, dropout=0.1):
        super().__init__()
        expand_dim = expand_dim or (dim * 2)
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

    def forward(self, x, adj, x_raw=None):
        # x: [B, N, dim], adj: [N, N]
        h = self.norm1(x)
        g_out = torch.matmul(adj, self.gcn(h))
        x = x + F.gelu(g_out)
        if x_raw is not None:
            x = x + 0.1 * x_raw
        h2 = self.norm2(x)
        out = x + self.mlp(h2)
        return out


# =========================================================================
# 2. Scaled Context Encoder & Scaled Velocity Field Network
# =========================================================================

class SpatioTemporalContextEncoder(nn.Module):
    """
    Scaled Spatio-Temporal Encoder with Jumping Knowledge (JK) connections.
    Encodes 20-dim input spatio-temporal features into station context vectors c.
    """
    def __init__(self, in_dim=20, hidden_dim=256, num_blocks=6, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.blocks = nn.ModuleList([
            UltraResidualGCNBlock(hidden_dim, expand_dim=hidden_dim * 2, dropout=dropout)
            for _ in range(num_blocks)
        ])
        # Jumping Knowledge across all layer depths
        self.jk_proj = nn.Sequential(
            nn.Linear(hidden_dim * (num_blocks + 1), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, feat, adj):
        # feat: [B, N, in_dim], adj: [N, N]
        h0 = self.in_proj(feat)
        h = h0
        intermediates = [h0]
        for block in self.blocks:
            h = block(h, adj, x_raw=h0)
            intermediates.append(h)
        jk = self.jk_proj(torch.cat(intermediates, dim=-1))
        return jk


class ScaledGraphVelocityFieldNet(nn.Module):
    """
    High-Capacity Velocity Field Network v_theta(x_t, t, c, Graph_A).
    Features:
      - Continuous time MLP with multi-layer frequency mapping.
      - Per-block time conditioning to maintain time gradient across all depths.
      - Deep residual GCN blocks with Jumping Knowledge.
      - Predicts instantaneous velocity u_t = dx_t / dt in R^{B, N, 2}.
    """
    def __init__(self, context_dim=256, hidden_dim=256, flow_dim=2, num_blocks=6, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.flow_dim = flow_dim
        self.num_blocks = num_blocks

        # Time MLP
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # State projection for [rentals, returns] log-flow
        self.state_proj = nn.Sequential(
            nn.Linear(flow_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Fusion projection: [x_t embedding + context c] -> hidden_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Per-block time modulation projections
        self.time_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_blocks)
        ])

        # Deep spatial velocity propagation blocks
        self.blocks = nn.ModuleList([
            UltraResidualGCNBlock(hidden_dim, expand_dim=hidden_dim * 2, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # Jumping Knowledge projection
        self.jk_proj = nn.Sequential(
            nn.Linear(hidden_dim * (num_blocks + 1), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Velocity output head
        self.out_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, flow_dim)
        )

    def forward(self, x_t, t, context, adj):
        """
        x_t: [B, N, 2] (flow state at continuous time t)
        t  : [B] (continuous time in [0, 1])
        context: [B, N, context_dim] (spatiotemporal context from encoder)
        adj: [N, N] (OD connectivity graph)
        Returns:
          v_theta: [B, N, 2] (velocity vector field dx_t / dt)
        """
        B, N, _ = x_t.shape
        t_emb = self.time_mlp(t)  # [B, hidden_dim]
        t_emb_expanded = t_emb.unsqueeze(1)  # [B, 1, hidden_dim]

        # Initial state embedding + context fusion
        h_state = self.state_proj(x_t) + t_emb_expanded
        h0 = self.fusion_proj(torch.cat([h_state, context], dim=-1))

        h = h0
        intermediates = [h0]
        for i, (block, time_proj) in enumerate(zip(self.blocks, self.time_projs)):
            t_mod = time_proj(t_emb).unsqueeze(1)
            h = block(h + t_mod, adj, x_raw=h0)
            intermediates.append(h)

        jk = self.jk_proj(torch.cat(intermediates, dim=-1))
        v_pred = self.out_head(jk)  # [B, N, 2]
        return v_pred


# =========================================================================
# 3. Scaled Continuous Flow Matching Framework & ODE Solvers
# =========================================================================

class SpatioTemporalFlowMatching(nn.Module):
    """
    Scaled Continuous Flow Matching (ST-CFM) Framework:
      - Optimal Transport displacement interpolation.
      - High-capacity context encoder + modulated velocity field network.
      - Solvers: Euler, Midpoint (RK2), RK4.
    """
    def __init__(self, in_dim=20, hidden_dim=256, flow_dim=2, num_blocks=6, dropout=0.1, sigma_prior=1.0):
        super().__init__()
        self.flow_dim = flow_dim
        self.sigma_prior = sigma_prior
        self.hidden_dim = hidden_dim

        self.encoder = SpatioTemporalContextEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout
        )
        self.v_net = ScaledGraphVelocityFieldNet(
            context_dim=hidden_dim,
            hidden_dim=hidden_dim,
            flow_dim=flow_dim,
            num_blocks=num_blocks,
            dropout=dropout
        )

    def compute_loss(self, feat, y_targets, adj):
        """
        Computes CFM Loss along Optimal Transport Displacement Paths:
          x_t = (1 - t) * x_0 + t * x_1
          u_t = x_1 - x_0
          L_CFM = || v_theta(x_t, t, c, adj) - u_t ||^2
        """
        B, N, _ = y_targets.shape
        device = y_targets.device

        # Data state: x_1 = log1p(y) in R^{B, N, 2}
        x_1 = torch.log1p(torch.clamp(y_targets, min=0.0))

        # Prior distribution: x_0 ~ N(0, sigma^2 * I)
        x_0 = torch.randn_like(x_1) * self.sigma_prior

        # Continuous time sampling: t ~ Uniform(0, 1)
        t = torch.rand(B, device=device)
        t_expand = t.view(B, 1, 1)

        # Straight displacement interpolation
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
        u_t = x_1 - x_0

        # Context representation
        context = self.encoder(feat, adj)

        # Vector field prediction
        v_pred = self.v_net(x_t, t, context, adj)

        # CFM Mean Squared Error loss
        loss = F.mse_loss(v_pred, u_t)
        return loss

    @torch.no_grad()
    def solve_ode(self, feat, adj, n_steps=10, solver='euler', x_0=None):
        """
        Numerical ODE integration from t=0 to t=1:
          dx_t / dt = v_theta(x_t, t, context, adj)
        solvers: 'euler', 'midpoint', 'rk4'
        """
        B, N, _ = feat.shape
        device = feat.device

        if x_0 is None:
            x_t = torch.zeros(B, N, self.flow_dim, device=device)
        else:
            x_t = x_0.clone()

        context = self.encoder(feat, adj)
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t_val = step * dt
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.float32)

            if solver == 'euler':
                v = self.v_net(x_t, t_batch, context, adj)
                x_t = x_t + dt * v

            elif solver == 'midpoint':
                # Runge-Kutta 2nd Order
                v1 = self.v_net(x_t, t_batch, context, adj)
                t_mid = torch.full((B,), t_val + 0.5 * dt, device=device, dtype=torch.float32)
                x_mid = x_t + 0.5 * dt * v1
                v2 = self.v_net(x_mid, t_mid, context, adj)
                x_t = x_t + dt * v2

            elif solver == 'rk4':
                # Classical 4th Order Runge-Kutta
                k1 = self.v_net(x_t, t_batch, context, adj)

                t_half = torch.full((B,), t_val + 0.5 * dt, device=device, dtype=torch.float32)
                k2 = self.v_net(x_t + 0.5 * dt * k1, t_half, context, adj)
                k3 = self.v_net(x_t + 0.5 * dt * k2, t_half, context, adj)

                t_full = torch.full((B,), t_val + dt, device=device, dtype=torch.float32)
                k4 = self.v_net(x_t + dt * k3, t_full, context, adj)

                x_t = x_t + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            else:
                raise ValueError(f"Unknown solver: {solver}")

        # Convert back from log-flow space to bike count: y = exp(x) - 1
        y_pred = torch.clamp(torch.expm1(x_t), min=0.0)
        return y_pred

    @torch.no_grad()
    def sample_scenarios(self, feat, adj, n_scenarios=20, n_steps=10, solver='midpoint'):
        """Samples K joint spatial scenarios for Monte Carlo uncertainty."""
        B, N, _ = feat.shape
        device = feat.device
        all_scenarios = []

        for _ in range(n_scenarios):
            x_0 = torch.randn(B, N, self.flow_dim, device=device) * self.sigma_prior
            y_s = self.solve_ode(feat, adj, n_steps=n_steps, solver=solver, x_0=x_0)
            all_scenarios.append(y_s.cpu())

        return torch.stack(all_scenarios, dim=0)  # [K, B, N, 2]


# =========================================================================
# 4. Evaluation Metrics
# =========================================================================

def compute_metrics(y_true, y_pred):
    """Calculates comprehensive metrics."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 1e-5)

    mae = float(np.abs(y_true - y_pred).mean())
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    wape = float(np.abs(y_true - y_pred).sum() / max(y_true.sum(), 1.0))

    zero_mask = (y_true == 0)
    zero_cnt = int(zero_mask.sum())
    mean_pred_at_zero = float(y_pred[zero_mask].mean()) if zero_cnt > 0 else 0.0
    pred_under_01_ratio = float((y_pred[zero_mask] < 0.1).sum() / max(zero_cnt, 1))

    p95_val = np.percentile(y_true, 95)
    peak_mask = (y_true >= max(p95_val, 3.0))
    peak_cnt = int(peak_mask.sum())
    peak_mae = float(np.abs(y_true[peak_mask] - y_pred[peak_mask]).mean()) if peak_cnt > 0 else 0.0

    ll = y_true * np.log(y_pred) - y_pred - scipy.special.gammaln(y_true + 1.0)
    mean_ll = float(ll.mean())

    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'wape': round(wape, 4),
        'poisson_ll': round(mean_ll, 4),
        'mean_at_zero': round(mean_pred_at_zero, 4),
        'zero_under_0_1_pct': round(pred_under_01_ratio * 100, 2),
        'peak_mae': round(peak_mae, 4)
    }


def compute_energy_score(y_true, scenarios):
    """Computes Multivariate Energy Score."""
    K, B, N = scenarios.shape
    y_exp = y_true[None, :, :]

    diff_y = np.linalg.norm(scenarios - y_exp, axis=-1)
    term1 = diff_y.mean(axis=0)

    scen_diffs = []
    for i in range(min(K, 10)):
        for j in range(i + 1, min(K, 10)):
            d = np.linalg.norm(scenarios[i] - scenarios[j], axis=-1)
            scen_diffs.append(d)
    term2 = np.mean(scen_diffs, axis=0) if scen_diffs else np.zeros_like(term1)

    es = (term1 - 0.5 * term2).mean()
    return float(es)


# =========================================================================
# 5. Training, Evaluation, and Step Sensitivity Pipeline
# =========================================================================

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="Scaled Spatio-Temporal Continuous Flow Matching (ST-CFM)")
    # Model Scale Arguments
    parser.add_argument('--hidden-dim', type=int, default=256, help="Hidden dimension for GCN & Flow Matching (128, 256, 384)")
    parser.add_argument('--num-blocks', type=int, default=6, help="Number of Residual GCN blocks (3, 6, 8)")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate")
    # Training Arguments
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=0.0012, help="Learning rate")
    parser.add_argument('--weight-decay', type=float, default=1e-4, help="Weight decay")
    # Inference Arguments
    parser.add_argument('--ode-steps', type=int, default=10, help="Number of ODE steps for inference")
    parser.add_argument('--solver', choices=['euler', 'midpoint', 'rk4'], default='midpoint', help="ODE solver")
    parser.add_argument('--fast', action='store_true', help="Fast smoke-test mode with fewer epochs/samples")
    parser.add_argument('--sample-scenarios', action='store_true', help="Run Monte Carlo scenario uncertainty sampling")
    parser.add_argument('--model-path', default='data/flow_matching_st_model.pt', help="Path to save/load model checkpoint")
    parser.add_argument('--eval-only', action='store_true', help="Skip training and evaluate existing checkpoint")
    parser.add_argument('--compare-steps', action='store_true', help="Compare different ODE integration step counts (3, 5, 10, 20)")
    parser.add_argument('--out', default='data/flow_matching_benchmark_results.json', help="Output results path")
    args = parser.parse_args()

    print("=" * 88, flush=True)
    print("      [Scaled Spatio-Temporal Continuous Flow Matching (ST-CFM)]", flush=True)
    print(f"      Scale Config: hidden_dim={args.hidden_dim}, num_blocks={args.num_blocks}, dropout={args.dropout}", flush=True)
    print("=" * 88, flush=True)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {dev}", flush=True)

    # Load dataset
    data = load_enriched_dataset(
        'data/processed/tashu_60min.npz',
        'data/clustering_stations.csv',
        'data/seoul_derived_tashu_features.npz'
    )
    adj = data['full_adj'].to(dev)
    N_sel = data['N_sel']

    # Initialize Scaled Continuous Flow Matching model
    model = SpatioTemporalFlowMatching(
        in_dim=20,
        hidden_dim=args.hidden_dim,
        flow_dim=2,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        sigma_prior=1.0
    ).to(dev)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Scaled ST-CFM Model Parameters: {n_params:,}", flush=True)

    batch_size = args.batch_size

    if args.eval_only:
        if not Path(args.model_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {args.model_path}")
        print(f"\n[Mode: Eval-Only] Loading trained checkpoint from: {args.model_path}", flush=True)
        model.load_state_dict(torch.load(args.model_path, map_location=dev))
    else:
        epochs = 3 if args.fast else args.epochs
        train_indices = np.where(data['train_mask'])[0]
        if args.fast:
            train_indices = train_indices[::2]
        n_train = len(train_indices)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        scaler = torch.amp.GradScaler('cuda', enabled=(dev.type == 'cuda'))

        print(f"\n[Training] Scaled Continuous Flow Matching (AMP FP16 Enabled)...", flush=True)
        print(f"           Epochs: {epochs}, Batch size: {batch_size}, Train samples: {n_train}", flush=True)
        t_start = time.time()
        best_loss = float('inf')

        for ep in range(1, epochs + 1):
            model.train()
            perm = np.random.permutation(train_indices)
            tot_loss = 0.0
            batches = 0
            t_ep0 = time.time()

            for i in range(0, n_train, batch_size):
                b_idx = perm[i:i + batch_size]
                if len(b_idx) < 4:
                    continue

                feat_b = prepare_deep_batch(data, b_idx).to(dev)  # [B, N, 20]
                y_r = torch.tensor(data['y_rent'][b_idx], dtype=torch.float32, device=dev)
                y_ret = torch.tensor(data['y_ret'][b_idx], dtype=torch.float32, device=dev)
                y_target = torch.stack([y_r, y_ret], dim=-1)  # [B, N, 2]

                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=(dev.type == 'cuda')):
                    loss = model.compute_loss(feat_b, y_target, adj)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()

                tot_loss += loss.item()
                batches += 1

            scheduler.step()
            ep_loss = tot_loss / max(batches, 1)
            elapsed = time.time() - t_ep0
            lr_cur = scheduler.get_last_lr()[0]
            print(f"   Epoch {ep:02d}/{epochs} | CFM Vector Field MSE: {ep_loss:.5f} | Time: {elapsed:.1f}s | LR: {lr_cur:.6f}", flush=True)

            if ep_loss < best_loss:
                best_loss = ep_loss
                torch.save(model.state_dict(), args.model_path)

        print(f"-> Training completed in {time.time() - t_start:.1f}s", flush=True)
        print(f"-> Best Model Checkpoint saved to: {args.model_path} (Best Loss: {best_loss:.5f})", flush=True)

    # =========================================================================
    # Evaluation on Strictly Held-Out Feb 2026 & Mar 2026
    # =========================================================================
    print("\n" + "=" * 88)
    print(f" [Evaluation] Numerical ODE Inference (Steps={args.ode_steps}, Solver={args.solver})")
    print("=" * 88)

    model.eval()
    periods = [
        ('feb_2026', '2026년 2월 (겨울 저수요)', data['feb_mask']),
        ('mar_2026', '2026년 3월 (봄 성수기/개학)', data['mar_mask'])
    ]

    all_results = {}

    for p_key, p_name, p_mask in periods:
        print(f"\n>> Evaluating on {p_name}...")
        t_indices = np.where(p_mask)[0]
        if args.fast:
            t_indices = t_indices[::4]

        # Gather ground truth
        y_true_rent = data['y_rent'][t_indices].reshape(-1)
        y_true_ret = data['y_ret'][t_indices].reshape(-1)

        # Baseline: Historical Smoothed Mean
        hist_pred_rent = data['hist_rent'][t_indices].reshape(-1)
        hist_pred_ret = data['hist_ret'][t_indices].reshape(-1)

        # ODE Point Forecast (Mean trajectory)
        t_eval_start = time.time()
        preds_rent_list = []
        preds_ret_list = []

        with torch.no_grad():
            for i in range(0, len(t_indices), batch_size):
                b_idx = t_indices[i:i + batch_size]
                feat_b = prepare_deep_batch(data, b_idx).to(dev)
                with torch.amp.autocast('cuda', enabled=(dev.type == 'cuda')):
                    y_pred = model.solve_ode(feat_b, adj, n_steps=args.ode_steps, solver=args.solver)
                preds_rent_list.append(y_pred[:, :, 0].cpu().numpy())
                preds_ret_list.append(y_pred[:, :, 1].cpu().numpy())

        pred_rent_ode = np.concatenate(preds_rent_list, axis=0).reshape(-1)
        pred_ret_ode = np.concatenate(preds_ret_list, axis=0).reshape(-1)
        eval_time = time.time() - t_eval_start
        ms_per_hour = (eval_time / len(t_indices)) * 1000.0

        # Compute metrics
        met_hist_rent = compute_metrics(y_true_rent, hist_pred_rent)
        met_ode_rent = compute_metrics(y_true_rent, pred_rent_ode)
        met_ode_ret = compute_metrics(y_true_ret, pred_ret_ode)

        print(f"   Inference Time: {eval_time:.2f}s ({ms_per_hour:.1f}ms per city-wide hour)", flush=True)
        print(f"\n   [대여 수요(Rentals) 성능 비교 - {p_name}]", flush=True)
        print(f"   {'모델':<32} | {'MAE':<7} | {'WAPE':<7} | {'Poisson LL':<10} | {'0대 예측':<8} | {'0대<0.1%':<10} | {'피크 MAE'}", flush=True)
        print("   " + "-" * 90, flush=True)
        print(f"   {'Historical Smoothed Mean':<32} | {met_hist_rent['mae']:>7.4f} | {met_hist_rent['wape']:>7.4f} | {met_hist_rent['poisson_ll']:>10.4f} | {met_hist_rent['mean_at_zero']:>8.4f} | {met_hist_rent['zero_under_0_1_pct']:>9.1f}% | {met_hist_rent['peak_mae']:>7.4f}", flush=True)
        print(f"   {'Continuous Flow Matching (CFM)':<32} | {met_ode_rent['mae']:>7.4f} | {met_ode_rent['wape']:>7.4f} | {met_ode_rent['poisson_ll']:>10.4f} | {met_ode_rent['mean_at_zero']:>8.4f} | {met_ode_rent['zero_under_0_1_pct']:>9.1f}% | {met_ode_rent['peak_mae']:>7.4f}", flush=True)

        # ODE Step Sensitivity Ablation (if requested)
        if args.compare_steps:
            print(f"\n   [ODE Integration Step Sensitivity (초고속 추론 특성 분석) - {p_name}]", flush=True)
            print(f"   {'Steps':<6} | {'Solver':<10} | {'ms / hr':<9} | {'MAE':<7} | {'WAPE':<7} | {'0대<0.1%':<10} | {'피크 MAE'}", flush=True)
            print("   " + "-" * 75, flush=True)
            for test_step in [3, 5, 10, 20]:
                for test_solver in ['euler', 'midpoint', 'rk4']:
                    t_s0 = time.time()
                    p_list = []
                    with torch.no_grad():
                        for i in range(0, len(t_indices), batch_size):
                            b_idx = t_indices[i:i + batch_size]
                            feat_b = prepare_deep_batch(data, b_idx).to(dev)
                            with torch.amp.autocast('cuda', enabled=(dev.type == 'cuda')):
                                yp = model.solve_ode(feat_b, adj, n_steps=test_step, solver=test_solver)
                            p_list.append(yp[:, :, 0].cpu().numpy())
                    p_arr = np.concatenate(p_list, axis=0).reshape(-1)
                    ms_h = ((time.time() - t_s0) / len(t_indices)) * 1000.0
                    m_step = compute_metrics(y_true_rent, p_arr)
                    print(f"   {test_step:<6d} | {test_solver:<10} | {ms_h:>7.1f}ms | {m_step['mae']:>7.4f} | {m_step['wape']:>7.4f} | {m_step['zero_under_0_1_pct']:>9.1f}% | {m_step['peak_mae']:>7.4f}", flush=True)

        period_data = {
            'rentals_historical_mean': met_hist_rent,
            'rentals_flow_matching': met_ode_rent,
            'returns_flow_matching': met_ode_ret,
            'inference_ms_per_hour': round(ms_per_hour, 2)
        }

        # =========================================================================
        # Scenario Generation & Uncertainty Evaluation (if requested)
        # =========================================================================
        if args.sample_scenarios:
            print(f"\n   [공동 확률 시나리오 생성 (Probabilistic Joint Scenarios)]")
            sub_idx = t_indices[:min(len(t_indices), 24)]
            sub_y = data['y_rent'][sub_idx]
            feat_sub = prepare_deep_batch(data, sub_idx).to(dev)

            n_scen = 10 if args.fast else 20
            t_scen0 = time.time()
            scenarios = model.sample_scenarios(feat_sub, adj, n_scenarios=n_scen, n_steps=args.ode_steps, solver=args.solver)
            scen_rent = scenarios[:, :, :, 0].numpy()

            es = compute_energy_score(sub_y, scen_rent)

            q10 = np.percentile(scen_rent, 10, axis=0)
            q90 = np.percentile(scen_rent, 90, axis=0)
            q05 = np.percentile(scen_rent, 5, axis=0)
            q95 = np.percentile(scen_rent, 95, axis=0)

            cov80 = float(((sub_y >= q10) & (sub_y <= q90)).mean() * 100.0)
            cov90 = float(((sub_y >= q05) & (sub_y <= q95)).mean() * 100.0)

            print(f"      Multivariate Energy Score (ES): {es:.4f} (Lower is better)")
            print(f"      80% Prediction Interval Coverage: {cov80:.1f}% (Nominal: 80.0%)")
            print(f"      90% Prediction Interval Coverage: {cov90:.1f}% (Nominal: 90.0%)")
            print(f"      Scenario Generation Time for {len(sub_idx)}h ({n_scen} samples): {time.time()-t_scen0:.2f}s")

            period_data['scenario_evaluation'] = {
                'energy_score': round(es, 4),
                'coverage_80': round(cov80, 2),
                'coverage_90': round(cov90, 2)
            }

        all_results[p_key] = period_data

    Path(args.out).write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nAll benchmark results saved to: {args.out}")


if __name__ == '__main__':
    main()
