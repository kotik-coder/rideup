import random
import math
import requests
import time
import json
from pathlib import Path
from shapely.geometry import Point
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
                json.dump({str(k): float(v) for k, v in self.cache.items()}, f)
        except Exception as e:
            print(f"Ошибка сохранения кэша: {e}", file=sys.stderr)

    def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        """Безопасное получение высоты с кэшированием"""
        cache_key = (round(float(lat), 5), round(float(lon), 5))
        
        if cache_key in self.cache:
            return float(self.cache[cache_key])

        time_since_last = time.time() - self.last_request_time
        if time_since_last < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - time_since_last)

        try:
            url = f"https://api.opentopodata.org/v1/srtm30m?locations={lat},{lon}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            elevation = float(data['results'][0]['elevation'])
            self.cache[cache_key] = elevation
            self._save_cache()
            self.last_request_time = time.time()
            return elevation
            
        except Exception as e:
            print(f"Ошибка API для ({lat:.5f}, {lon:.5f}): {str(e)}", file=sys.stderr)
            return None

elevation_cache = ElevationCache()

def safe_get_elevation(lat: float, lon: float) -> float:
    """Получение высоты с гарантией возврата числа"""
    elevation = elevation_cache.get_elevation(float(lat), float(lon))
    return elevation if elevation is not None else random.uniform(150, 250)

def get_boundary_point(polygon, bounds) -> Tuple[float, float]:
    """Генерация точки на границе полигона"""
    for _ in range(100):
        side = random.choice(['north', 'south', 'east', 'west'])
        if side == 'north':
            lat, lon = float(bounds[3]), random.uniform(float(bounds[0]), float(bounds[2]))
        elif side == 'south':
            lat, lon = float(bounds[1]), random.uniform(float(bounds[0]), float(bounds[2]))
        elif side == 'east':
            lat, lon = random.uniform(float(bounds[1]), float(bounds[3])), float(bounds[2])
        else:
            lat, lon = random.uniform(float(bounds[1]), float(bounds[3])), float(bounds[0])
        
        if polygon.boundary.distance(Point(float(lon), float(lat))) < 0.0001:
            return (float(lat), float(lon))
    return (float(bounds[1]), float(bounds[0]))

def generate_nearby_point(reference_point, polygon, max_distance, bounds) -> Optional[Tuple[float, float]]:
    """Генерация точки в пределах расстояния"""
    ref_lat, ref_lon = float(reference_point[0]), float(reference_point[1])
    for _ in range(100):
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(100, float(max_distance))
        delta_lat = distance * math.cos(angle) / 111320
        delta_lon = distance * math.sin(angle) / (111320 * math.cos(math.radians(ref_lat)))
        
        new_lat = ref_lat + delta_lat
        new_lon = ref_lon + delta_lon
        
        if (float(bounds[1]) <= new_lat <= float(bounds[3]) and 
            float(bounds[0]) <= new_lon <= float(bounds[2]) and
            polygon.contains(Point(float(new_lon), float(new_lat)))):
            return (float(new_lat), float(new_lon))
    return None

def get_landscape_description(elevation: float, delta: float) -> str:
    """Описание ландшафта на основе высоты и перепада"""
    elevation = float(elevation)
    delta = float(delta)
    
    if elevation < 160:
        base = "Низина у ручья"
    elif elevation > 200:
        base = "Лесная возвышенность"
    else:
        base = "Смешанный лес"
    
    if delta > 15:
        return f"{base}, крутой подъем"
    elif delta < -15:
        return f"{base}, крутой спуск"
    return f"{base}, ровный участок"