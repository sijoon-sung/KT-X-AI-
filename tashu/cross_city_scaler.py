"""Cross-City Shared Bike Scaling and Alignment Module.

Provides robust mathematical scaling methods to align heterogeneous bike-sharing datasets
across cities with vastly different spatial scales, station densities, and trip volumes
(e.g., Daejeon Tashu vs. Seoul Ttareungi).

Four Core Scalers:
1. DiurnalSimplexScaler: Normalizes 24h temporal curves to unit simplex (sum = 1.0).
2. PerStationScaler: Scales demand by active station count or rack capacity (Turnover Rate).
3. BoundedFlowScaler: Maps asymmetric directional flows into [-1, 1] Net Flow index.
4. RelativeSpatialScaler: Projects local coordinates to canonical city-centered km offsets.
"""

from typing import Dict, Tuple, Optional
import numpy as np


class DiurnalSimplexScaler:
    """Scales hourly demand profiles onto the 24-dimensional probability simplex.
    Decouples absolute trip counts from diurnal human mobility patterns.
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def fit_transform(self, hourly_24h: np.ndarray) -> np.ndarray:
        """Args:
            hourly_24h: array of shape (24,) or (24, N) or (T, 24)
        Returns:
            Normalized array where sum along hour dimension equals 1.0
        """
        arr = np.asarray(hourly_24h, dtype=np.float64)
        if arr.ndim == 1:
            total = np.sum(arr)
            return arr / max(total, self.eps)
        elif arr.shape[0] == 24:
            totals = np.sum(arr, axis=0, keepdims=True)
            return arr / np.maximum(totals, self.eps)
        elif arr.shape[1] == 24:
            totals = np.sum(arr, axis=1, keepdims=True)
            return arr / np.maximum(totals, self.eps)
        else:
            raise ValueError(f"Expected 24-hour dimension, got shape {arr.shape}")


class PerStationScaler:
    """Normalizes macroscopic system demand by the number of active stations or docks,
    yielding 'Trips Per Station-Hour' (Turnover Intensity).
    """
    def __init__(self, station_count: int, capacity_sum: Optional[float] = None):
        self.station_count = max(1, station_count)
        self.capacity_sum = capacity_sum

    def transform(self, hourly_series: np.ndarray) -> np.ndarray:
        return np.asarray(hourly_series, dtype=np.float64) / self.station_count

    def transform_turnover(self, hourly_series: np.ndarray) -> np.ndarray:
        if self.capacity_sum is None or self.capacity_sum <= 0:
            return self.transform(hourly_series)
        return np.asarray(hourly_series, dtype=np.float64) / self.capacity_sum


class BoundedFlowScaler:
    """Normalizes directional rental/return asymmetry into a scale-free Net Flow index in [-1, +1].
    -1: Pure Origin/Outflow (residential morning departure)
    +1: Pure Destination/Inflow (office/transit morning arrival)
     0: Balanced Flow
    """
    @staticmethod
    def net_flow(returns: np.ndarray, rentals: np.ndarray, eps: float = 0.1) -> np.ndarray:
        returns = np.asarray(returns, dtype=np.float64)
        rentals = np.asarray(rentals, dtype=np.float64)
        return (returns - rentals) / np.maximum(returns + rentals, eps)


class RelativeSpatialScaler:
    """Centers and scales lat/lon into canonical relative kilometers from city centroid.
    Normalizes by the 95th percentile city radius to make spatial dispersion scale-invariant.
    """
    def __init__(self):
        self.center_lat = 0.0
        self.center_lon = 0.0
        self.radius_km_95 = 1.0

    def fit(self, lat: np.ndarray, lon: np.ndarray):
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        valid = ~np.isnan(lat) & ~np.isnan(lon)
        self.center_lat = np.median(lat[valid])
        self.center_lon = np.median(lon[valid])

        dy_km = (lat[valid] - self.center_lat) * 111.0
        dx_km = (lon[valid] - self.center_lon) * (111.0 * np.cos(np.radians(self.center_lat)))
        r = np.sqrt(dx_km**2 + dy_km**2)
        self.radius_km_95 = float(np.percentile(r, 95))
        if self.radius_km_95 <= 0.1:
            self.radius_km_95 = 1.0
        return self

    def transform(self, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        dy_km = (lat - self.center_lat) * 111.0
        dx_km = (lon - self.center_lon) * (111.0 * np.cos(np.radians(self.center_lat)))
        norm_r = np.sqrt(dx_km**2 + dy_km**2) / self.radius_km_95
        return dx_km, dy_km, norm_r
