from typing import List, Dict
from dataclasses import dataclass
import math

MAX_DISTANCE_KM = 3.0

#@dataclass
class GeoPoint:
    lat : float
    lon : float
    
    def __init__(self, lat, lon): 
        self.lat = lat
        self.lon = lon

    def distance_to(self, point : "GeoPoint"):
        """Вычисляет расстояние между двумя точками в метрах (упрощенная формула)"""
        lat2, lon2 = point.lat, point.lon
        return math.sqrt((lat2-self.lat)**2 + (lon2-self.lon)**2) * 111320  # Примерно 111 км на градус

@dataclass
class Route:
    name: str
    points: List[GeoPoint]  # (lat, lon)
    elevations: List[float]
    descriptions: List[str]
    source: str = "local"

    def to_map_format(self) -> Dict:
        return {
            "points": self.points,
            "names": [self.name] * len(self.points),
            "elevations": self.elevations,
            "descriptions": self.descriptions,
            "metadata": {
                "name": self.name,
                "source": self.source,
                "url": ""
            }
        }

    def _point_in_bounds(self, bounds : List[float], point: GeoPoint) -> bool:
        # Изменено: Доступ к lat и lon через атрибуты объекта GeoPoint
        lat = point.lat
        lon = point.lon
        
        # --- Исправлено ---
        # Правильная проверка границ: широта с границами по широте, долгота с границами по долготе.
        # bounds = [min_lon, min_lat, max_lon, max_lat]
        if (bounds[1] <= lat <= bounds[3] and
            bounds[0] <= lon <= bounds[2]):
            return True
        
        # Если точка вне прямоугольника, проверяем расстояние до границ
        # Вычисляем расстояния до каждой границы в километрах
        dist_north  = max(0, lat - bounds[3]) * 111.32 # max_lat
        dist_south  = max(0, bounds[1] - lat) * 111.32 # min_lat
        dist_east   = max(0, lon - bounds[2]) * 111.32 * math.cos(math.radians(lat)) # max_lon
        dist_west   = max(0, bounds[0] - lon) * 111.32 * math.cos(math.radians(lat)) # min_lon
        
        # Находим минимальное расстояние до границы
        min_distance = min(dist_north, dist_south, dist_east, dist_west)
        
        return min_distance <= MAX_DISTANCE_KM

    def is_valid_route(self, bounds : List[float]) -> bool:
        """Проверяет что маршрут полностью находится в допустимой зоне"""
        if not self.points:
            return False
            
        if not all(self._point_in_bounds(bounds, point) for point in self.points):
            return False
            
        return True