import math
import requests
import time
import json
from pathlib import Path
from typing import Tuple, List, Dict
import sys

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

class ElevationCache:
    def __init__(self):
        self.cache = self._load_cache()
        self.last_request_time = 0

    def _load_cache(self) -> Dict[Tuple[float, float], float]:
        try:
            if CACHE_FILE.exists():
                with open(CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)
                    # Convert string keys back to tuples of floats
                    return {
                        tuple(map(float, k.strip("()").split(','))): float(v)
                        for k, v in cache_data.items()
                    }
        except Exception as e:
            print(f"Ошибка загрузки кэша: {e}", file=sys.stderr)
        return {}

    def _save_cache(self):
        try:
            with open(CACHE_FILE, 'w') as f:
                # Convert tuple keys to string for JSON serialization
                json.dump({str(k): float(v) for k, v in self.cache.items()}, f)
        except Exception as e:
            print(f"Ошибка сохранения кэша: {e}", file=sys.stderr)

elevation_cache = ElevationCache()

def safe_get_elevation(lat: float, lon: float) -> float:
    key = (lat, lon)
    if key in elevation_cache.cache:
        return elevation_cache.cache[key]

    # Implement rate limiting
    current_time = time.time()
    if current_time - elevation_cache.last_request_time < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - (current_time - elevation_cache.last_request_time))
    
    try:
        url = f"https://api.opentopodata.org/v1/ned10m?locations={lat},{lon}"
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        elevation = data['results'][0]['elevation']
        elevation_cache.cache[key] = elevation
        elevation_cache.last_request_time = time.time()
        elevation_cache._save_cache() # Save cache after each new elevation fetch
        return elevation
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса высоты для ({lat},{lon}): {e}", file=sys.stderr)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Ошибка парсинга ответа или структуры данных для ({lat},{lon}): {e}", file=sys.stderr)
    return 0.0 # Return a default elevation in case of an error

def point_to_segment_projection_and_distance(
    point_lat: float, point_lon: float,
    seg_start_lat: float, seg_start_lon: float,
    seg_end_lat: float, seg_end_lon: float
) -> Tuple[float, float, float, float]: # <--- Changed return type to include t
    """
    Calculates the shortest distance from a point to a line segment
    and returns the coordinates of the projected point on the segment,
    as well as the 't' value (normalized position along the segment).
    Uses a simplified Cartesian-like distance for projection, then converts
    the final small distance to meters.

    Args:
        point_lat, point_lon: Coordinates of the photo checkpoint.
        seg_start_lat, seg_start_lon: Coordinates of the start of the route segment.
        seg_end_lat, seg_end_lon: Coordinates of the end of the route segment.

    Returns:
        Tuple[float, float, float, float]:
        (distance_in_meters, projected_lat, projected_lon, t_value)
        where t_value is between 0.0 and 1.0 if projection is on segment,
        <0.0 if closer to start, >1.0 if closer to end.
    """

    lat_to_m = 111320
    lon_to_m = 111320 * math.cos(math.radians(point_lat))

    Px, Py = point_lon * lon_to_m, point_lat * lat_to_m
    Ax, Ay = seg_start_lon * lon_to_m, seg_start_lat * lat_to_m
    Bx, By = seg_end_lon * lon_to_m, seg_end_lat * lat_to_m

    ABx, ABy = Bx - Ax, By - Ay
    APx, APy = Px - Ax, Py - Ay

    len_sq_AB = ABx**2 + ABy**2

    if len_sq_AB == 0: # Segment is a point
        dist = math.sqrt(APx**2 + APy**2)
        # For a point segment, t can be considered 0.0 (at A)
        return dist, seg_start_lat, seg_start_lon, 0.0 # <--- Changed return

    t = (APx * ABx + APy * ABy) / len_sq_AB

    if t < 0.0:
        projected_x, projected_y = Ax, Ay
    elif t > 1.0:
        projected_x, projected_y = Bx, By
    else:
        projected_x = Ax + t * ABx
        projected_y = Ay + t * ABy

    dx = Px - projected_x
    dy = Py - projected_y
    distance_in_meters = math.sqrt(dx**2 + dy**2)

    projected_lon = projected_x / lon_to_m
    projected_lat = projected_y / lat_to_m

    return distance_in_meters, projected_lat, projected_lon, t

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