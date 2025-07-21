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

TILE_SIZE = 512 # Pixel size of a single map tile. Common values are 256 or 512.
EARTH_CIRCUMFERENCE = 40075016.686 # In meters
DEGREES_PER_RADIAN = 180 / math.pi

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
    

def bounds_to_zoom(bounds, map_width_px, map_height_px, padding=0.0):
    """
    Calculates the optimal Mapbox zoom level to fit a given geographical bounding box.

    Args:
        bounds (dict): A dictionary with 'min_lon', 'max_lon', 'min_lat', 'max_lat'.
        map_width_px (int): The width of the map figure in pixels.
        map_height_px (int): The height of the map figure in pixels.
        padding (float): A factor to zoom out slightly for better framing (e.g., 0.1 for 10% padding).

    Returns:
        float: The calculated zoom level.
    """
    # 1. Calculate the required zoom level for the longitude span (width)
    lon_span = abs(bounds['max_lon'] - bounds['min_lon'])
    
    # Handle the case where the longitude span is zero
    if lon_span == 0:
      # If there's only one point, we can't calculate a zoom level from a span.
      # Default to a reasonable zoom level, or handle as needed.
      # Here, we'll just focus on latitude calculation.
      zoom_lon = 20 # A high zoom level for a single point
    else:
      zoom_lon = math.log2(map_width_px * 360 / (lon_span * TILE_SIZE))

    # 2. Calculate the required zoom level for the latitude span (height)
    # This is more complex due to the Mercator projection's distortion.
    def lat_to_mercator_y(lat):
        # Converts latitude to a 'y' coordinate in the Mercator projection
        lat_rad = math.radians(lat)
        return math.log(math.tan(math.pi / 4 + lat_rad / 2))

    y_merc_min = lat_to_mercator_y(bounds['min_lat'])
    y_merc_max = lat_to_mercator_y(bounds['max_lat'])
    lat_merc_span = abs(y_merc_max - y_merc_min)

    # Handle the case where the latitude span is zero
    if lat_merc_span == 0:
        zoom_lat = 20 # A high zoom level for a single point
    else:
        zoom_lat = math.log2(map_height_px / (lat_merc_span / (2 * math.pi) * TILE_SIZE))

    # 3. Choose the smaller of the two zoom levels to ensure the entire box fits
    zoom = min(zoom_lon, zoom_lat)
    
    # 4. Apply padding by slightly zooming out
    final_zoom = zoom * (1 - padding)

    return final_zoom