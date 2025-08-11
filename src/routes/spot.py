from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import osmnx as ox
from shapely import MultiPolygon, Polygon

# Assuming these are available in your project structure
from src.iio.weather_advice import WeatherAdvisor
from src.iio.terrain_loader import TerrainLoader
from src.routes.route_processor import ProcessedRoute, RouteProcessor
from src.ui.map_helpers import print_step
from src.iio.gpx_loader import LocalGPXLoader # Assuming LocalGPXLoader is within gpx_loader
from src.routes.route import Route
from src.iio.spot_photo import SpotPhoto
from src.routes.track import Track
from src.routes.terrain_and_weather import *

@dataclass
class RatingSystem:
    """Central configuration for all trail parameters including features and gradients"""
    # Gradient thresholds
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

    # Feature parameters
    feature_parameters: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "ROLLER": {
                "min_length": 50,
                "max_length": 500,
                "gradient_range": (-0.05, 0.05),
                "difficulty_impact": 1.5
            },
            "SWITCHBACK": {
                "min_length": 15,
                "max_length": 30,
                "gradient_range": (-0.25, -0.15),
                "difficulty_impact": 2.0
            },
            "TECHNICAL_DESCENT": {
                "min_length": 5,
                "max_length": 500,
                "gradient_range": (-float('inf'), -0.15),
                "difficulty_impact": 3.0
            },
            "TECHNICAL_ASCENT": {
                "min_length": 5,
                "max_length": 200,
                "gradient_range": (0.15, float('inf')),
                "difficulty_impact": 2.5
            },
            "FLOW_DESCENT": {
                "min_length": 50,
                "max_length": 500,
                "gradient_range": (-0.12, -0.05),
                "wavelength_range": (10, 50),
                "difficulty_impact": 1.2
            },
            "KICKER": {
                "min_length": 1,
                "max_length": 10,
                "gradient_range": (-0.3, -0.15),
                "difficulty_impact": 2.5
            },
            "DROP": {
                "min_length": 1,
                "max_length": 8,
                "gradient_range": (-float('inf'), -0.25),
                "difficulty_impact": 3.5
            }
        }
    )

    # Feature compatibility with gradient types
    feature_compatibility: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "ASCENT": ["TECHNICAL_ASCENT"],
            "DESCENT": ["TECHNICAL_DESCENT", "FLOW_DESCENT", "SWITCHBACK"],
            "STEEP_ASCENT": ["TECHNICAL_ASCENT"],
            "STEEP_DESCENT": ["TECHNICAL_DESCENT", "DROP", "KICKER"],
            "FLAT": []
        }
    )

    # Segment length parameters
    min_segment_length: float = 50
    min_steep_length: float = 10
    step_feature_max_length: float = 15
    
    # Wavelength parameters
    wavelength_clustering_eps: float = 0.5
    wavelength_match_tolerance: float = 0.3
    flow_wavelength_min: float = 10
    flow_wavelength_max: float = 50

    def validate_config(self) -> List[str]:
        """Check configuration consistency"""
        errors = []
        
        # Check gradient thresholds
        prev_max = None
        for name, (min_val, max_val) in sorted(self.gradient_thresholds.items()):
            if prev_max is not None and min_val < prev_max:
                errors.append(f"Gradient threshold overlap: {name}")
            prev_max = max_val
            
        # Check feature parameters
        for feature, params in self.feature_parameters.items():
            if params['min_length'] > params['max_length']:
                errors.append(f"Invalid length range for {feature}")
                
        return errors

    def get_feature_config(self, feature_type: str) -> Dict[str, Any]:
        """Get configuration for a specific feature type"""
        return self.feature_parameters.get(feature_type, {})

    def get_compatible_features(self, gradient_type: str) -> List[str]:
        """Get features that can occur on this gradient type"""
        return self.feature_compatibility.get(gradient_type, [])

    def is_feature_compatible(self, feature_type: str, gradient_type: str) -> bool:
        """Check if a feature type is compatible with a gradient type"""
        return feature_type in self.feature_compatibility.get(gradient_type, [])

    @classmethod
    def create(cls, custom_config: Optional[Dict] = None) -> 'RatingSystem':
        """Factory method with enhanced feature support"""
        system = cls()
        
        if custom_config:
            # Handle nested feature configurations
            for key, value in custom_config.items():
                if key == "feature_parameters" and isinstance(value, dict):
                    system.feature_parameters.update(value)
                elif key == "feature_compatibility" and isinstance(value, dict):
                    system.feature_compatibility.update(value)
                elif hasattr(system, key):
                    setattr(system, key, value)
        
        return system

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
    weather: WeatherData
    weather_api_key : str

    def __init__(self, geostring: str, system_config: Optional[Dict] = None, weather_api_key: Optional[str] = None):
        self.system = RatingSystem.create(system_config)
        self.name = geostring
        self.geostring = geostring
        self._photos_loaded = False
        self.local_photos = []
        self.routes = []
        self.tracks = []
        self.geometry = []
        self.processed_routes = {}
        self._get_bounds(geostring)
        self.load_valid_routes_and_tracks()
        self.local_photos = SpotPhoto.load_local_photos(self.bounds)
        self.terrain = None
        self._load_terrain_data()
        self.weather = None

        self.api_key = weather_api_key
        self.query_weather()
        
        print_step("Spot", f"Spot '{self.name}' initialized with bounds: {self.bounds}")

    def query_weather(self) -> Optional[WeatherData]:
         # Initialize weather (will try Roshydromet first, then OpenWeatherMap if key provided)
        self.weather = WeatherAdvisor.get_current_weather(self.bounds, self.api_key)
        
        # Print weather advice if available
        if self.weather:
            self.advice = self.get_weather_advice()
            print_step("Weather Advice", self.advice)

        return self.weather

    def get_traction(self):
        if self.weather:
            return self.terrain.get_adjusted_traction(self.weather)
        else:
            return self.terrain.traction_score

    def load_weather_data(self, api_key: Optional[str] = None) -> Optional[WeatherData]:
        """
        Load current weather conditions for this spot.
        Args:
            api_key: Optional OpenWeatherMap API key (uses Roshydromet if None)
        Returns:
            WeatherData object if successful, None otherwise
        """
        self.weather = WeatherAdvisor.get_current_weather(self.bounds, api_key)
        if self.weather:
            print_step("Weather", 
                f"Current conditions: {self.weather.condition}\n"
                f"Temperature: {self.weather.temperature}°C\n"
                f"Wind: {self.weather.wind_speed} m/s\n"
            )
        return self.weather
        
    def get_weather_advice(self) -> str:
        """
        Get riding advice based on current weather and terrain conditions.
        Returns formatted string with emoji indicators.
        """
        return WeatherAdvisor.generate_riding_advice(self.weather, self.terrain)
                        
    def _recommend_bike_type(self, surface: str, traction: float) -> str:
        """
        Recommends a bike type with key specs. Uses British English.
        Fits compact UIs (target: ~80–100 characters).
        """
        surface = surface.lower().strip()
        
        # Surface-specific (highest priority)
        if "sand" in surface:
            return "Fat bike"
        if "mud" in surface or "clay" in surface:
            return "Enduro/Downhill"
        if "rock" in surface or "scree" in surface or "ledge" in surface:
            return "Enduro" if traction > 0.5 else "DH bike only"

        # Generic surfaces – use traction
        if traction > 0.75:
            return "XC or Gravel"
        elif traction > 0.6:
            return "XC or Trail/All-Mountain"
        elif traction > 0.45:
            return "Trail/All-Mountain"
        else:
            return "Enduro"
        
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