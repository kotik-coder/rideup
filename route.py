# route.py
from dataclasses import dataclass
from functools import reduce
from typing import List, Optional
import math

from folium import Polygon
import shapely

MAX_DISTANCE_KM = 3.0

@dataclass
class GeoPoint:
    """Static geographic point"""
    lat: float
    lon: float
    elevation: float = 0.0
    
    def to_dict(self) -> dict:
        """Converts the GeoPoint object to a dictionary for serialization."""
        return {
            'lat': self.lat,
            'lon': self.lon,
            'elevation': self.elevation
        }

    def distance_to(self, point: "GeoPoint"):
        R = 6371000 # Radius of Earth in meters
        
        lat1_rad = math.radians(self.lat)
        lon1_rad = math.radians(self.lon)
        lat2_rad = math.radians(point.lat)
        lon2_rad = math.radians(point.lon)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c
        return distance # Distance in meters
    
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

@dataclass
class Route:
    """Static route definition"""
    name: str
    points: List[GeoPoint]
    elevations: List[float]
    descriptions: List[str]
    total_distance: Optional[float]

    def _point_within_polygon(self, bounds, point: GeoPoint) -> bool:        
        shapely_point = shapely.Point(point.lon, point.lat)        
        return shapely_point.within(bounds)

    def is_valid_route(self, bounds : List[float]) -> bool:
        """Проверяет что маршрут полностью находится в допустимой зоне"""
        if not self.points:
            return False
                
        within = reduce(lambda count, i: count + self._point_within_polygon(bounds, i), self.points, 0)
        fraction_within = within / ( (float) (len(self.points)) )                    
        return fraction_within > 0.5