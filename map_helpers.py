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

def print_step(prefix : str, message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] [{prefix}] [{level}] {message}")

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



def get_landscape_description(
    current_elevation: float,
    segment_net_elevation_change: float,
    segment_elevation_gain: float,
    segment_elevation_loss: float,
    segment_distance: float
) -> str:
    """
    Описание ландшафта на основе текущей высоты и характеристик сегмента.
    
    Args:
        current_elevation: Текущая высота в точке чекпоинта.
        segment_net_elevation_change: Чистое изменение высоты по сегменту (конец - начало).
        segment_elevation_gain: Общий набор высоты по сегменту.
        segment_elevation_loss: Общий спуск высоты по сегменту.
        segment_distance: Длина сегмента в метрах.
    """
    desc = ""
    
    # 1. Общее описание высоты точки
    if current_elevation < 100:
        desc += "Низменность. "
    elif 100 <= current_elevation < 200:
        desc += "Равнинная местность. "
    else:
        desc += "Возвышенность. "

    # 2. Описание перепада высот по сегменту
    if segment_distance > 0: # Avoid division by zero
        # Consider average gradient for more nuanced description
        avg_gradient_gain = (segment_elevation_gain / segment_distance) * 100 if segment_distance > 0 else 0
        avg_gradient_loss = (segment_elevation_loss / segment_distance) * 100 if segment_distance > 0 else 0

        # Heuristics for "ravines" or "undulating"
        # If both gain and loss are significant relative to segment length and net change
        if segment_elevation_gain > 15 and segment_elevation_loss > 15 and \
           abs(segment_net_elevation_change) < (segment_elevation_gain + segment_elevation_loss) * 0.5:
            desc += "Пересеченная местность с оврагами или чередованием подъемов и спусков. "
        elif segment_elevation_gain > 20:
            desc += "Значительный подъем. "
        elif segment_elevation_loss > 20:
            desc += "Значительный спуск. "
        elif segment_net_elevation_change > 5:
            desc += "Легкий подъем. "
        elif segment_net_elevation_change < -5:
            desc += "Легкий спуск. "
        else:
            desc += "Пологий участок. "
    else:
        desc += "Короткий, ровный участок. "
            
    return desc

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