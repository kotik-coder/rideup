import math
from pathlib import Path
from typing import List
from datetime import datetime

CACHE_FILE = Path("elevation_cache.json")
REQUEST_DELAY = 1.0  # Задержка между запросами в секундах

DEBUG = True

TILE_SIZE = 512 # Pixel size of a single map tile. Common values are 256 or 512.

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