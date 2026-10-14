"""
neural_zic_model.py
-------------------
Zero-Inflated Censored Neural Operator (ZIC-Net)
Combines:
1. Structural Zero Gate: p_i(t) in [0, 1] (Probability that station i has zero true demand)
2. Censored Latent Demand Intensity: lambda_i(t) > 0 (Macro flux M(t) * Boltzmann pi_i(t))
3. Physics-Preserving Dynamic Neural Advection Operator: P_OD(t) ensuring exact mass conservation sum(mu) == sum(lambda).
4. Censored Log-Likelihood Loss accommodating stockouts and zero-demand inflation.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kinetic_flux_demand import MacroFluxHead, BoltzmannSpatialHead, nb_logpmf
from neural_flux_operator import DynamicNeuralAdvection, build_topk_sparse_prior

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ZICNet(nn.Module):
    """
    Zero-Inflated Censored Neural Operator for Urban Bike-Sharing Flux.
    """
    def __init__(self, n_stations, od_matrix, base_spatial_prior=None, k=32, d_model=32):
        super().__init__()
        self.n_stations = n_stations
        self.k = k
        self.d_model = d_model

        # 1. Macro Flux Head: M(t)
        self.macro_head = MacroFluxHead()

        # 2. Boltzmann Spatial Potential Head: pi_i(t)
        self.spatial_head = BoltzmannSpatialHead(n_stations, base_spatial_prior)

        # 3. Dynamic Neural Advection Operator: P_OD(t)
        top_k_indices, log_prior = build_topk_sparse_prior(od_matrix, k=k)
        self.advection_op = DynamicNeuralAdvection(n_stations, top_k_indices, log_prior, d_model=d_model)

        # 4. Structural Zero Gate Head: p_i(t) in (0, 1)
        # Determines if a station has structural zero demand regardless of inventory.
        self.station_zero_emb = nn.Embedding(n_stations, 16)
        self.zero_gate_mlp = nn.Sequential(
            nn.Linear(16 + 24, 32), # 16 (station) + 24 (hour 12 + dow 6 + month 6)
            nn.GELU(),
            nn.Linear(32, 1)
        )
        # Initialize gate bias to lean towards active (p ~ 0.1) initially
        nn.init.constant_(self.zero_gate_mlp[-1].bias, -2.0)

        # Context embeddings for zero gate
        self.hour_emb = nn.Embedding(24, 12)
        self.weekday_emb = nn.Embedding(7, 6)
        self.month_emb = nn.Embedding(12, 6)

        # Dispersion parameters for Negative-Binomial observation
        self.log_r = nn.Parameter(torch.tensor([2.0, 2.0])) # [r_rent, r_return]

    def compute_zero_gate(self, hour, weekday, month):
        """
        Computes structural zero probability p_i(t) for all stations.
        Returns:
            p_zero: [B, N] in (0, 1)
        """
        B, N = hour.shape[0], self.n_stations
        ctx = torch.cat([self.hour_emb(hour), self.weekday_emb(weekday),
                         self.month_emb(month)], dim=-1) # [B, 24]
        ctx_expanded = ctx.unsqueeze(1).expand(B, N, -1) # [B, N, 24]
        st_emb = self.station_zero_emb.weight.unsqueeze(0).expand(B, N, -1) # [B, N, 16]

        feat = torch.cat([st_emb, ctx_expanded], dim=-1) # [B, N, 40]
        logits = self.zero_gate_mlp(feat).squeeze(-1) # [B, N]
        return torch.sigmoid(logits)

    def forward(self, hour, weekday, month, station_mask=None):
        """
        Forward pass producing:
        - lambda_rent: [B, N] true uncensored latent rental demand rate
        - mu_return: [B, N] dynamically transported return arrival rate
        - p_zero: [B, N] structural zero probability
        - M: [B] city-wide macro demand
        - pi: [B, N] spatial distribution
        - P_dyn: [B, N, K] transition matrix
        """
        B = hour.shape[0]

        # A. Latent Demand Generation (Physics/Boltzmann)
        M = self.macro_head(hour, weekday, month) # [B]
        pi, V = self.spatial_head(hour, weekday, month, station_mask) # [B, N]
        lambda_rent = M.unsqueeze(1) * pi # [B, N]

        # B. Structural Zero Gate
        p_zero = self.compute_zero_gate(hour, weekday, month) # [B, N]
        if station_mask is not None:
            # Inactive stations have p_zero forced to near 1.0
            p_zero = torch.where(station_mask, p_zero, torch.ones_like(p_zero) * 0.999)

        # Effective active demand: lambda_active = (1 - p_zero) * lambda_rent
        lambda_effective = (1.0 - p_zero) * lambda_rent

        # C. Mass-Preserving Dynamic Advection
        mu_return, P_dyn = self.advection_op(lambda_effective, hour, weekday, month)

        return {
            'lambda_rent': lambda_rent,
            'lambda_effective': lambda_effective,
            'mu_return': mu_return,
            'p_zero': p_zero,
            'M': M,
            'pi': pi,
            'P_dyn': P_dyn
        }


def zic_loss_function(pred, actual_rentals, actual_returns, exposure, censored, active_mask):
    """
    Zero-Inflated Censored Log-Likelihood Loss.
    
    Accommodates:
    - Zero-inflation gate p_zero
    - Censored observation where y_obs = 0 can be either structural zero OR stockout starvation.
    - Negative-Binomial dispersion for rentals and returns.
    """
    lambda_eff = pred['lambda_effective'] # [B, N]
    mu_ret = pred['mu_return']           # [B, N]
    p_zero = pred['p_zero']               # [B, N]

    # Effective rate adjusted by exposure (fraction of hour with stock > 0)
    tau = torch.clamp(exposure, min=0.01, max=1.0)
    rate_rent = lambda_eff * tau
    rate_ret = torch.clamp(mu_ret, min=1e-5)

    # 1. Zero-inflated likelihood for rentals
    # If y == 0: P(Y=0) = p_zero + (1 - p_zero) * exp(-rate_rent)
    # If y > 0:  P(Y=y) = (1 - p_zero) * Poisson(y; rate_rent)
    y_rent = actual_rentals.float()
    is_zero = (y_rent == 0)

    # Log P(Y=0)
    prob_zero = p_zero + (1.0 - p_zero) * torch.exp(-rate_rent)
    log_p_zero = torch.log(torch.clamp(prob_zero, min=1e-7))

    # Log P(Y=y > 0)
    # log(Poisson(y; lambda)) = y*log(lambda) - lambda - lgamma(y+1)
    log_p_pos = (
        torch.log(torch.clamp(1.0 - p_zero, min=1e-7)) +
        y_rent * torch.log(torch.clamp(rate_rent, min=1e-7)) -
        rate_rent -
        torch.lgamma(y_rent + 1.0)
    )

    ll_rent = torch.where(is_zero, log_p_zero, log_p_pos)

    # 2. Return loss: Poisson likelihood for returns
    y_ret = actual_returns.float()
    ll_ret = (
        y_ret * torch.log(rate_ret) -
        rate_ret -
        torch.lgamma(y_ret + 1.0)
    )

    # Mask inactive stations
    mask = active_mask.float()
    loss_rent = -(ll_rent * mask).sum() / (mask.sum() + 1e-6)
    loss_ret = -(ll_ret * mask).sum() / (mask.sum() + 1e-6)

    # Regularization on structural zero gate: L1 sparsity so model doesn't label everything zero
    loss_gate_reg = p_zero.mean() * 0.05

    total_loss = loss_rent + 0.5 * loss_ret + loss_gate_reg
    return total_loss, {
        'loss_rent': float(loss_rent.item()),
        'loss_ret': float(loss_ret.item()),
        'loss_gate_reg': float(loss_gate_reg.item())
    }
