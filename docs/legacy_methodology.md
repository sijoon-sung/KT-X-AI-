# Multimodal Transit Continuous Flow Matching (Transit-CFM)
## Spatio-Temporal Continuous Flow Matching for Urban Micro-Mobility & Transit Policy Interventions

This repository contains the core methodology implementation of **Multimodal Transit Continuous Flow Matching (Transit-CFM)**, a graph-informed continuous generative framework designed for city-wide bike-sharing demand modeling and counterfactual transit policy simulations.

---

### 1. Proposed Methodology (`tashu/multimodal_transit_flow_matching.py`)

The proposed method models joint spatial velocity fields along Optimal Transport (OT) displacement trajectories, conditioned on multimodal transit connectivity and meteorological forcing:

1. **Optimal Transport Continuous Flow Matching (OT-CFM):**
   - Straight displacement paths: $\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\mathbf{x}_1$ for $t \in [0, 1]$.
   - Target velocity field: $\mathbf{u}_t = \mathbf{x}_1 - \mathbf{x}_0$.
   - Objective: $\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1} \| v_\theta(\mathbf{x}_t, t, \mathbf{C}, \tilde{\mathbf{A}}) - (\mathbf{x}_1 - \mathbf{x}_0) \|^2$.

2. **Multimodal Transit & External Conditioning ($\mathbf{C} \in \mathbb{R}^{B \times N \times 36}$):**
   - **Metro Line 1 Coupling (22 Stations):** Geodesic distance decay $e^{-d / 500\text{m}}$ and commuter rush waves (08h inflow / 18h outflow).
   - **Bus Transit Coupling (159 Hubs):** Last-mile feeder index $e^{-d / 300\text{m}}$ and transfer wave intensity.
   - **6-Class Urban Functional Typologies:** One-hot embeddings for Subway Hub, Bus Transit, Residential Origin, Campus/Research, Civic/Commercial, and Park/Leisure.
   - **Hourly Dynamic Weather:** Real-time temperature, precipitation, wind speed, relative humidity, and rain-transit suppression interactions.

3. **High-Capacity Residual GCN Backbone:**
   - 6 Residual GCN Blocks with LayerNorm and 2x FFN Expansion (dim 512).
   - Continuous sinusoidal time embedding with per-block AdaLN/FiLM modulation.
   - Dense Jumping Knowledge (JK) multi-scale skip connections.

4. **Numerical ODE Inference & What-If Policy Engine:**
   - Supports Euler, Midpoint (Runge-Kutta 2nd order), and Classical RK4 solvers.
   - Solves forward and counterfactual trajectories: $\hat{\mathbf{x}}_1 = \mathbf{x}_0 + \int_0^1 v_\theta(\mathbf{x}_t, t, \mathbf{C}, \tilde{\mathbf{A}}) dt$.

---

### 2. Method Code Structure

```
├── tashu/
│   ├── multimodal_transit_flow_matching.py   # [Main Proposed Method] Full Transit-CFM pipeline
│   ├── flow_matching_st.py                   # [Foundation Method] Scaled ST-CFM architecture
│   └── data/
│       ├── multimodal_transit_benchmark_results.json # Benchmark metrics & policy simulation results
│       └── flow_matching_benchmark_results.json      # Pure ST-CFM benchmark results
├── .gitignore                                # Strict exclusion of *.pt checkpoints & raw datasets
└── README.md                                 # Methodology documentation
```

---

### 3. How to Run the Methodology Code

#### Requirements
- Python >= 3.10
- PyTorch >= 2.0 (CUDA enabled)
- NumPy, SciPy, LightGBM, XGBoost

#### Execution Command
```bash
cd tashu

# Run full 10-epoch training and 2026 held-out evaluation
python multimodal_transit_flow_matching.py --epochs 10 --batch-size 16 --solver midpoint --run-policy-sim

# Fast smoke-test mode (3 epochs, quick verification)
python multimodal_transit_flow_matching.py --fast --run-policy-sim
```

---

### 4. Empirical Evaluation Summary (Held-Out 2026 Data)

| Period | Model | MAE | WAPE | Zero-Demand Pred | Peak MAE ($y \ge 3$) | Rainy Hours MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Feb 2026** (Winter) | Historical Smoothed Mean | 0.5281 | 1.1622 | 0.3731 | 2.0950 | 0.5120 |
| | **Multimodal Transit-CFM (Proposed)** | **0.3395** | **0.7471** | **0.0787** | **2.1458** | **0.1171** |
| **Mar 2026** (Spring) | Historical Smoothed Mean | 0.5264 | 0.9311 | 0.2778 | 2.5899 | 0.5230 |
| | **Multimodal Transit-CFM (Proposed)** | **0.3809** | **0.6738** | **0.0811** | **2.1855** | **0.0568** |

- **Inference Latency:** 12.7 ms per city-wide hour (1,283 stations simultaneously on RTX 3080).
- **Counterfactual Policy Simulation:**
  - *Subway Hub 50% Bike Supply Expansion:* +3.60% station catchment inflow.
  - *Commuter Transit Disruption Shock:* +10.74% subway catchment surge, +11.18% city-wide last-mile absorption.