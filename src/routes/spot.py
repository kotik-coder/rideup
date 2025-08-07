import requests
import tempfile
import rasterio
from rasterstats import zonal_stats
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import osmnx as ox
from shapely import MultiPolygon, Polygon

# Assuming these are available in your project structure
from src.iio.terrain_loader import TerrainLoader
from src.routes.route_processor import ProcessedRoute, RouteProcessor
from src.ui.map_helpers import print_step
from src.iio.gpx_loader import LocalGPXLoader # Assuming LocalGPXLoader is within gpx_loader
from src.routes.route import Route
from src.iio.spot_photo import SpotPhoto
from src.routes.track import Track

@dataclass
class RatingSystem:
    """Central configuration for all trail parameters"""
    # Gradient thresholds (now using direct defaults)
    gradient_thresholds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "ASCENT": (0.01, 0.10),
            "DESCENT": (-0.10, -0.01),
            "STEEP_ASCENT": (0.10, float('inf')),
            "STEEP_DESCENT": (-float('inf'), -0.10),
            "FLAT": (-0.01, 0.01)
        }
    )
    
    # Difficulty thresholds
    difficulty_thresholds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "GREEN": (0, 2.0), 
            "BLUE": (2.0, 7.0),
            "BLACK": (7.0, 15.0),
            "DOUBLE_BLACK": (15.0, float('inf'))
        }
    )

    # Other parameters...
    min_segment_length: float = 50
    min_steep_length: float = 10
    step_feature_max_length: float = 15
    wavelength_clustering_eps: float = 0.5
    wavelength_match_tolerance: float = 0.3
    flow_wavelength_min: float = 10
    flow_wavelength_max: float = 50

    @classmethod
    def create(cls, custom_config: Optional[Dict] = None) -> 'RatingSystem':
        """Factory method for creating configured systems"""
        system = cls()  # Let dataclass handle initialization
        
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(system, key):
                    setattr(system, key, value)
        
        return system

@dataclass 
class TerrainAnalysis:
    """Stores terrain type information for a spot"""
    surface_types: Dict[str, float]  # Surface type distribution (e.g., {'dirt': 0.6, 'gravel': 0.3})
    landcover_types: Dict[str, float]  # ESA WorldCover classification
    dominant_trail_surface: str
    traction_score: float  # 0-1 estimate of overall traction

@dataclass
class Spot:
    name: str
    geostring: str
    bounds: List[float]
    geometry : List[Tuple[float,float]]
    
    local_photos: List[SpotPhoto]
    
    routes: List[Route]
    processed_routes : Dict[int, ProcessedRoute]
    tracks: List[Track]  # Tracks now contain their route reference
    polygon: any
    terrain: TerrainAnalysis

    def __init__(self, geostring: str, system_config: Optional[Dict] = None):
        self.system = RatingSystem.create(system_config)  # Add this line
        self.name = geostring
        self.geostring = geostring
        self._photos_loaded = False # Internal flag to prevent redundant photo loading
        self.local_photos = []
        self.routes = []
        self.tracks = []
        self.geometry = []
        self.processed_routes = {}
        self._get_bounds(geostring)
        self.load_valid_routes_and_tracks()
        self.local_photos = SpotPhoto.load_local_photos(self.bounds)
        self.terrain = None  # Will hold TerrainAnalysis object
        self._load_terrain_data()  # New terrain loading
        print_step("Spot", f"Spot '{self.name}' initialized with bounds: {self.bounds}")
                    
    def _recommend_bike_type(self, surface: str, traction: float) -> str:
        """Suggest suitable bike type based on terrain analysis.
        Args:
            surface: Dominant surface type (e.g., 'gravel', 'rock').
            traction: Composite score (0-1) from terrain weights.
        Returns:
            str: Bike type recommendation.
        """
        if surface == "sand":
            return "Fat Bike (4.5\"+ tires or wider)"
        elif surface == "mud":
            return "Fat Bike or Enduro (mud tires)"
        elif traction > 0.75:
            return "Road/Gravel Bike"  # Smooth asphalt/concrete
        elif traction > 0.6:
            return "XC/Hardtail"       # Compacted gravel/dirt
        elif traction > 0.45:
            return "Trail/All-Mountain"  # Rough but rideable
        elif surface == "rock":
            return "Enduro/DH (2.4\"+ tires)"
        else:
            return "Enduro/DH (full suspension)"
        
    def _load_terrain_data(self):
        """Simplified terrain loading using TerrainLoader"""
        self.terrain = TerrainLoader.load_terrain(self.geostring, self.polygon)
        
        # Print formatted results
        if self.terrain.surface_types:
            surface_report = "\n".join(
                f"- {k}: {v:.1%}" 
                for k, v in sorted(
                    self.terrain.surface_types.items(),
                    key=lambda x: -x[1]
                )
            )
            print_step("Terrain",
                f"Trail Surfaces:\n{surface_report}\n"
                f"Dominant: {self.terrain.dominant_surface}\n"
                f"Traction: {self.terrain.traction_score:.0%}\n"
                f"Recommended bike type: {self._recommend_bike_type(self.terrain.dominant_surface, self.terrain.traction_score)}"
            )

    def get_processed_route(self, rp : RouteProcessor, route : Route, selected_index : int) -> ProcessedRoute:
        route_dict = self.processed_routes
        
        if selected_index in route_dict:     
            processed_route = self.processed_routes[selected_index]
        else:                
            route_dict[selected_index] = rp.process_route(route)
            processed_route = route_dict[selected_index]
            
        return processed_route
        
    def load_valid_routes_and_tracks(self):        
        print_step("SpotLoader", "Loading routes and tracks...")
        
        # Collect all (Route, List[Track]) pairs from all GPX files
        loaded_route_track_pairs = LocalGPXLoader.load_all_gpx()
        
        self.routes = [] # Reset to store only valid Route objects
        self.tracks = [] # Reset to be a flattened list of all valid Track objects

        for route, associated_tracks in loaded_route_track_pairs:
            if route.is_valid_route(self.polygon):
                self.routes.append(route)
                # Set the route reference for each track
                for track in associated_tracks:
                    track.route = route
                self.tracks.extend(associated_tracks)

        print_step("SpotLoader", f"Found {len(self.routes)} valid routes for '{self.name}'")
        print_step("SpotLoader", f"Found {len(self.tracks)} tracks for '{self.name}'")        

        return self.routes

    def _get_bounds(self, geostring: str):
        """
        Retrieves the geographical bounds of the forest using OSMnx.
        Returns a list [min_lon, min_lat, max_lon, max_lat].
        """
        print_step(prefix="Spot", message=f"Getting bounds for '{geostring}'...")
        try:
            gdf         = ox.geocode_to_gdf(geostring)
            self.bounds = [float(x) for x in gdf.total_bounds]
            print_step("Spot", f"Bounds obtained: {self.bounds}")        
                            
            self.polygon = gdf.iloc[0].geometry
            
            #if it's just a polygon
            if isinstance(self.polygon, Polygon):                
                lon, lat = self.polygon.exterior.coords.xy
                self.geometry = list(zip(lon, lat))                        
            #if it contains multiple polygons
            if isinstance(self.polygon, MultiPolygon):  
                for polygon in self.polygon.geoms:
                    lon, lat = polygon.exterior.coords.xy
                    self.geometry.extend(list(zip(lon, lat)))
            
        except Exception as e:
            print_step("Error", f"Failed to get bounds for '{geostring}': {e}", level="ERROR")