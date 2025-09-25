# route.py
from dataclasses import dataclass, field
from functools import reduce
from typing import List, Optional, Tuple
import math
import shapely

@dataclass(frozen=True)
class GeoPoint:
    lat: float
    lon: float
    elevation: float = 0.0

    # Mutable field — not part of the "identity"
    distance_from_origin: float = field(default=0.0, compare=False, hash=False)    
    
    def set_distance_from_origin(self, dist: float) -> None:
        # Use object.__setattr__ to bypass frozen restriction
        object.__setattr__(self, 'distance_from_origin', dist)
    
    def to_dict(self) -> dict:
        """Converts the GeoPoint object to a dictionary for serialization."""
        return {
            'lat': self.lat,
            'lon': self.lon,
            'elevation': self.elevation
        }
        
    def bearing_to(self, point: "GeoPoint") -> float:
        """
        Calculate initial bearing from this point to another point.
        Returns bearing in degrees (0-360), where 0° is North.
        """
        lat1 = math.radians(self.lat)
        lon1 = math.radians(self.lon)
        lat2 = math.radians(point.lat)
        lon2 = math.radians(point.lon)

        dlon = lon2 - lon1

        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

        bearing = math.degrees(math.atan2(y, x))
        return (bearing + 360) % 360
    
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

@dataclass(frozen=True)
class Route:
    """Static, immutable route definition"""
    name: str
    points: Tuple[GeoPoint, ...]          # ← tuple, not list
    elevations: Tuple[float, ...]         # ← tuple
    descriptions: Tuple[str, ...]         # ← tuple
    total_distance: Optional[float] = None

    def _point_within_polygon(self, polygon: shapely.geometry.Polygon, point: GeoPoint) -> bool:
        from shapely.geometry import Point
        shapely_point = Point(point.lon, point.lat)
        return polygon.contains(shapely_point)

    def is_valid_route(self, polygon: shapely.geometry.Polygon) -> bool:
        if not self.points:
            return False
        within_count = sum(
            1 for point in self.points if self._point_within_polygon(polygon, point)
        )
        return (within_count / len(self.points)) > 0.5