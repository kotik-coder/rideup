import math
from pathlib import Path
from typing import List

from datetime import datetime 

CACHE_FILE = Path("elevation_cache.json")
REQUEST_DELAY = 1.0  # Задержка между запросами в секундах

DEBUG = True

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