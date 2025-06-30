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
    
    def bearing_to(self, other_point: "GeoPoint") -> float:
        """
        Calculates the initial bearing (direction) from this GeoPoint to another GeoPoint.
        Args:
            other_point (GeoPoint): The destination GeoPoint.
        Returns:
            float: Bearing in degrees (0-360) from true North.
        """
        lat1_rad = math.radians(self.lat)
        lon1_rad = math.radians(self.lon)
        lat2_rad = math.radians(other_point.lat)
        lon2_rad = math.radians(other_point.lon)

        delta_lon = lon2_rad - lon1_rad

        x = math.cos(lat2_rad) * math.sin(delta_lon)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)

        initial_bearing_rad = math.atan2(x, y)
        initial_bearing_deg = math.degrees(initial_bearing_rad)

        # Normalize to 0-360 degrees
        return (initial_bearing_deg + 360) % 360

    def point_at_distance_and_bearing(self, distance_m: float, bearing_deg: float) -> "GeoPoint":
        """
        Calculates a new GeoPoint given this GeoPoint, a distance, and a bearing.
        Args:
            distance_m (float): Distance in meters.
            bearing_deg (float): Bearing in degrees from true North (0-360).
        Returns:
            GeoPoint: A new GeoPoint instance.
        """
        R = 6371000  # Radius of Earth in meters

        lat_rad = math.radians(self.lat)
        lon_rad = math.radians(self.lon)
        bearing_rad = math.radians(bearing_deg)

        new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(distance_m / R) +
                                math.cos(lat_rad) * math.sin(distance_m / R) * math.cos(bearing_rad))

        new_lon_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance_m / R) * math.cos(lat_rad),
                                         math.cos(distance_m / R) - math.sin(lat_rad) * math.sin(new_lat_rad))

        new_lat = math.degrees(new_lat_rad)
        new_lon = math.degrees(new_lon_rad)

        return GeoPoint(new_lat, new_lon)

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