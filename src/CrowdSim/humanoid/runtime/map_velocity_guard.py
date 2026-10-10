"""Separate a map comfort cost from an explicitly chosen physical guard gap."""
import math

def map_velocity_hazard(future_clearance, current_clearance, comfort_m, tracking_m, *, physical_threshold_m=None):
    if physical_threshold_m is not None and (type(physical_threshold_m) not in (float,int)
        or not math.isfinite(physical_threshold_m) or not 0<=physical_threshold_m<=1.):
        raise ValueError('Explicit finite physical map guard gap in[0,1]m required')
    threshold=comfort_m+tracking_m if physical_threshold_m is None else physical_threshold_m
    return (future_clearance<threshold)&(future_clearance<current_clearance-.005)
