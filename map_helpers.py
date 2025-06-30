import random
import math
import requests
import time
import json
from pathlib import Path
from shapely.geometry import Point, Polygon # Import Polygon as well for type hinting
from typing import Tuple, List, Optional, Dict
import sys

from datetime import datetime

CACHE_FILE = Path("elevation_cache.json")
REQUEST_DELAY = 1.0  # Задержка между запросами в секундах

def print_step(prefix : str, message: str):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] [{prefix}] {message}")

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

def get_boundary_point(polygon: Polygon, bounds: List[float]) -> Tuple[float, float]:
    """
    Генерирует случайную точку на границе полигона леса.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # Try to find a point on the actual boundary
    for _ in range(100): # Attempts to find a point on the boundary
        # Randomly choose a side (top, bottom, left, right)
        side = random.choice(['horizontal', 'vertical'])
        if side == 'horizontal':
            lat = random.uniform(min_lat, max_lat)
            lon = random.choice([min_lon, max_lon])
        else:
            lon = random.uniform(min_lon, max_lon)
            lat = random.choice([min_lat, max_lat])
        
        # Check if this point is close to the polygon's boundary
        point_candidate = Point(float(lon), float(lat))
        if polygon.boundary.distance(point_candidate) < 0.0001: # Check if it's very close to boundary
            return (float(lat), float(lon))
            
    # Fallback: if no point on boundary is found, return a corner or center of bounds
    return (float(min_lat), float(min_lon)) # Default to bottom-left corner


def generate_nearby_point(reference_point: Tuple[float, float], polygon: Polygon, max_distance: float, bounds: List[float]) -> Optional[Tuple[float, float]]:
    """Генерация точки в пределах расстояния от reference_point внутри полигона."""
    ref_lat, ref_lon = float(reference_point[0]), float(reference_point[1])
    
    # Convert meters to degrees approximately (at equator)
    # 1 degree of latitude approx 111,320 meters
    # 1 degree of longitude approx (111,320 * cos(latitude)) meters
    
    for _ in range(100): # Max attempts to find a valid point
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(100, float(max_distance)) # Distance in meters
        
        # Calculate delta_lat and delta_lon
        delta_lat = distance * math.cos(angle) / 111320
        delta_lon = distance * math.sin(angle) / (111320 * math.cos(math.radians(ref_lat)))
        
        new_lat = ref_lat + delta_lat
        new_lon = ref_lon + delta_lon
        
        # Check if the new point is within the bounding box and inside the polygon
        if (float(bounds[1]) <= new_lat <= float(bounds[3]) and 
            float(bounds[0]) <= new_lon <= float(bounds[2]) and
            polygon.contains(Point(float(new_lon), float(new_lat)))):
            return (float(new_lat), float(new_lon))
            
    return None # Return None if no valid point is found after attempts

def get_boundary_near_point(polygon: Polygon, bounds: List[float], reference_point: Tuple[float, float], max_distance_from_ref: float) -> Optional[Tuple[float, float]]:
    """
    Генерирует точку на границе полигона, которая находится в пределах
    заданного расстояния от reference_point.
    """
    ref_lat, ref_lon = float(reference_point[0]), float(reference_point[1])
    min_lon, min_lat, max_lon, max_lat = bounds

    # Approximate degrees per meter at the reference latitude
    deg_per_meter_lat = 1 / 111320
    deg_per_meter_lon = 1 / (111320 * math.cos(math.radians(ref_lat)))

    search_radius_deg_lat = max_distance_from_ref * deg_per_meter_lat
    search_radius_deg_lon = max_distance_from_ref * deg_per_meter_lon

    # Define a smaller bounding box around the reference point within which to search for boundary points
    search_min_lat = max(min_lat, ref_lat - search_radius_deg_lat)
    search_max_lat = min(max_lat, ref_lat + search_radius_deg_lat)
    search_min_lon = max(min_lon, ref_lon - search_radius_deg_lon)
    search_max_lon = min(max_lon, ref_lon + search_radius_deg_lon)

    # Number of attempts to find a suitable point
    for _ in range(200): # Increased attempts for better chance
        # Randomly pick a point on the perimeter of the *search* bounding box
        side = random.choice(['top', 'bottom', 'left', 'right'])
        
        if side == 'top':
            lat_candidate = search_max_lat
            lon_candidate = random.uniform(search_min_lon, search_max_lon)
        elif side == 'bottom':
            lat_candidate = search_min_lat
            lon_candidate = random.uniform(search_min_lon, search_max_lon)
        elif side == 'left':
            lon_candidate = search_min_lon
            lat_candidate = random.uniform(search_min_lat, search_max_lat)
        else: # 'right'
            lon_candidate = search_max_lon
            lat_candidate = random.uniform(search_min_lat, search_max_lat)

        current_point = Point(float(lon_candidate), float(lat_candidate))
        
        # Check if this point is near the actual forest boundary AND within the polygon
        # The 0.0002 threshold is a small tolerance for floating point comparisons / proximity to boundary
        if polygon.boundary.distance(current_point) < 0.0002 and polygon.contains(current_point):
            return (float(lat_candidate), float(lon_candidate))
            
    # If no point is found near the reference within max_distance, fallback to a general boundary point
    # This ensures a finish point is always found, even if not ideal
    return get_boundary_point(polygon, bounds)


def get_landscape_description(elevation: float, delta: float) -> str:
    """Описание ландшафта на основе высоты и перепада"""
    elevation = float(elevation)
    delta = float(delta)
    
    desc = ""
    if elevation < 100:
        desc += "Низменность. "
    elif 100 <= elevation < 200:
        desc += "Равнинная местность. "
    else:
        desc += "Возвышенность. "

    if delta > 10:
        desc += "Значительный подъем."
    elif delta < -10:
        desc += "Значительный спуск."
    elif 3 <= delta <= 10:
        desc += "Легкий подъем."
    elif -10 <= delta <= -3:
        desc += "Легкий спуск."
    else:
        desc += "Пологий участок."
    
    return desc