import math
from pathlib import Path
from typing import List

from datetime import datetime

import numpy as np 

CACHE_FILE = Path("elevation_cache.json")
REQUEST_DELAY = 1.0  # Задержка между запросами в секундах

DEBUG = True

# Define WGS-84 ellipsoid constants for high accuracy
WGS84_A = 6378137.0  # Semi-major axis in meters
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_E2 = WGS84_F * (2 - WGS84_F)  # Square of first eccentricity

def geodesic_integrand(t, interp_lat, interp_lon, dlat_dt, dlon_dt):
    """
    The function to be integrated to find the arc length.
    This corresponds to the ds/dt term in the mathematical explanation.
    """
    # Get latitude and its derivative at parameter t
    lat_deg = interp_lat(t)
    dlat_dt_deg = dlat_dt(t)
    # Get longitude derivative at parameter t
    dlon_dt_deg = dlon_dt(t)

    # Convert degrees (from spline) to radians for trigonometric functions
    lat_rad = np.radians(lat_deg)
    dlat_dt_rad = np.radians(dlat_dt_deg)
    dlon_dt_rad = np.radians(dlon_dt_deg)

    # Calculate radii of curvature M and N
    sin_lat_sq = np.sin(lat_rad)**2
    den = np.sqrt(1 - WGS84_E2 * sin_lat_sq)
    
    M = (WGS84_A * (1 - WGS84_E2)) / (den**3)  # Meridional radius
    N = WGS84_A / den  # Prime vertical radius
    
    # Calculate the term under the square root
    integrand_val_sq = (M * dlat_dt_rad)**2 + (N * np.cos(lat_rad) * dlon_dt_rad)**2
    
    return np.sqrt(integrand_val_sq)

def print_step(prefix : str, message: str, level: str = "INFO"):    
    if(DEBUG):
        timestamp = datetime.now().strftime('%H:%M:%S')
        msg = f"[{timestamp}] [{prefix}] [{level}] {message}"
        if(str == 'ERROR'): 
            from colorama import Fore, Style            
            msg = Fore.RED + msg + Style.RESET_ALL
        print(msg)

def expanded_bounds(bounds : List[float], distance_km) -> List[float]: 
    lon_min, lat_min, lon_max, lat_max = bounds
    
    km_per_deg_lat = 111.0
        
    # For longitude, it depends on latitude
    avg_lat = (lat_min + lat_max) / 2
    km_per_deg_lon = 111.0 * math.cos(math.radians(avg_lat))
    
    # Calculate degree offsets
    lat_offset = distance_km / km_per_deg_lat
    lon_offset = distance_km / km_per_deg_lon
    
    return [
        lon_min - lon_offset,
        lat_min - lat_offset,
        lon_max + lon_offset,
        lat_max + lat_offset
    ]