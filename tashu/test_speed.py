import numpy as np
import time
import xgboost as xgb
import lightgbm as lgb

print("XGBoost version:", xgb.__version__)
print("LightGBM version:", lgb.__version__)

# Quick test of 500k rows
N = 500000
p = 10
X = np.random.randn(N, p).astype(np.float32)
y = np.random.poisson(lam=1.5, size=N).astype(np.float32)

t0 = time.time()
m_lgb = lgb.LGBMRegressor(objective='poisson', n_estimators=50, num_leaves=31, n_jobs=-1, verbose=-1)
m_lgb.fit(X, y)
t_lgb = time.time() - t0
print(f"LightGBM 500k rows trained in: {t_lgb:.2f}s")

t0 = time.time()
m_xgb = xgb.XGBRegressor(objective='count:poisson', n_estimators=50, max_depth=6, n_jobs=-1, tree_method='hist')
m_xgb.fit(X, y)
t_xgb = time.time() - t0
print(f"XGBoost (hist) 500k rows trained in: {t_xgb:.2f}s")
