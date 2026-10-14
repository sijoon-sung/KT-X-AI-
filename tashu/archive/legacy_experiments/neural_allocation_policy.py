"""
neural_allocation_policy.py
--------------------------
Differentiable Neural Allocation Policy Network (DNAP).
Learns the optimal 6 AM bicycle pre-allocation vector by backpropagating directly
through a fully differentiable 15-minute micro-timeline commute simulator.

Key features:
1. Exploration Floor Guarantee: Automatically secures >= 1 bike for all non-structural-zero active stations.
2. Differentiable Softmax Water-Filling: Allocates remaining city bike budget to high-flux sinks.
3. Fully Differentiable 15-Minute Micro-Timeline: End-to-end gradient flow from unmet demand back to policy weights!
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NeuralPolicyAllocator(nn.Module):
    """
    Differentiable Neural Policy Network for 6 AM Bicycle Pre-Allocation.
    """
    def __init__(self, n_stations, hidden_dim=64):
        super().__init__()
        self.n_stations = n_stations
        self.hidden_dim = hidden_dim

        # Per-station feature encoder: [log(1 + demand), p_zero, 1 - p_zero, log(cap), is_active] -> 5 features
        self.encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Temperature parameter for allocation sharpness
        self.log_tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, dem_4h, p_zero, capacities, is_active, total_bikes=3070.0):
        """
        Computes the continuous differentiable allocation vector.
        
        Args:
            dem_4h: [N] predicted 4-hour peak latent demand
            p_zero: [N] predicted structural zero probability
            capacities: [N] capacity cap (e.g. 60 or 25)
            is_active: [N] boolean active mask
            total_bikes: scalar float
            
        Returns:
            alloc: [N] allocated bicycle counts (sum(alloc) == total_bikes)
            floor: [N] guaranteed exploration floor
        """
        N = self.n_stations
        
        # 1. Prepare features
        log_dem = torch.log1p(torch.clamp(dem_4h, min=0.0))
        p_z = torch.clamp(p_zero, 0.0, 1.0)
        p_act = 1.0 - p_z
        log_cap = torch.log1p(torch.clamp(capacities, min=1.0))
        act_float = is_active.float()
        
        feat = torch.stack([log_dem, p_z, p_act, log_cap, act_float], dim=-1) # [N, 5]
        
        # 2. Compute priority score per station
        logits = self.encoder(feat).squeeze(-1) # [N]
        
        # Inactive stations and structural zeroes (p_zero > 0.6) get heavily masked logits
        mask_eligible = is_active & (p_zero < 0.6)
        logits = torch.where(mask_eligible, logits, torch.full_like(logits, -1e6))
        
        # 3. Compute Guaranteed Exploration Floor
        # Active stations with p_zero < 0.6 get at least 1 bike
        floor = torch.where(mask_eligible, torch.ones_like(logits), torch.zeros_like(logits))
        floor_sum = floor.sum()
        
        # Remaining budget after exploration floor
        rem_budget = torch.clamp(total_bikes - floor_sum, min=1.0)
        
        # 4. Softmax allocation of remaining budget
        tau = torch.exp(self.log_tau).clamp(0.1, 5.0)
        weights = F.softmax(logits / tau, dim=0) # [N]
        
        # Raw allocation
        surplus_alloc = weights * rem_budget
        alloc = floor + surplus_alloc
        
        # Differentiable soft capping at capacities
        alloc = torch.minimum(alloc, capacities)
        
        # Rescale slightly if cap was exceeded to maintain exact total_bikes
        alloc = alloc * (total_bikes / torch.clamp(alloc.sum(), min=1.0))
        
        return alloc, floor


def simulate_differentiable_micro_timeline(start_inv, rent_4h, od_matrix_norm, num_steps=16):
    """
    Fully differentiable PyTorch implementation of the 15-minute micro-timeline simulation.
    Gradients flow directly from total_unmet back to start_inv!
    
    Args:
        start_inv: [N] tensor of initial inventory at 06:00
        rent_4h: [N] tensor of total 4-hour demand
        od_matrix_norm: [N, N] tensor of transition probabilities (sum_j P_ij = 1)
        num_steps: 16 quarters (4 hours)
        
    Returns:
        total_unmet: scalar tensor (differentiable)
        total_served: scalar tensor (differentiable)
    """
    N = start_inv.shape[0]
    inv = start_inv.clone()
    
    # 15-minute demand profile weights
    quarter_weights = torch.tensor([
        0.015, 0.025, 0.025, 0.035, # 06:00 - 07:00 (10%)
        0.050, 0.070, 0.080, 0.100, # 07:00 - 08:00 (30%)
        0.110, 0.110, 0.100, 0.080, # 08:00 - 09:00 (40%)
        0.060, 0.050, 0.050, 0.040  # 09:00 - 10:00 (20%)
    ], device=start_inv.device, dtype=torch.float32)
    quarter_weights = quarter_weights / quarter_weights.sum()

    total_unmet = torch.tensor(0.0, device=start_inv.device)
    total_served = torch.tensor(0.0, device=start_inv.device)
    in_transit = torch.zeros(N, device=start_inv.device)

    for t in range(num_steps):
        # 1. Returning bikes from previous step arrive!
        inv = inv + in_transit
        in_transit = torch.zeros(N, device=start_inv.device)
        
        # 2. Demand in this 15-minute quarter
        step_demand = rent_4h * quarter_weights[t]
        
        # 3. Fulfilled rentals vs unmet demand (differentiable via torch.minimum & relu)
        served = torch.minimum(inv, step_demand)
        unmet = F.relu(step_demand - inv)
        
        inv = inv - served
        total_served = total_served + served.sum()
        total_unmet = total_unmet + unmet.sum()
        
        # 4. Advection: served bikes move according to OD matrix and arrive in next step
        # in_transit = served @ OD
        in_transit = torch.matmul(served.unsqueeze(0), od_matrix_norm).squeeze(0)

    return total_unmet, total_served
