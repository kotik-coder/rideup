# spot.py
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from matplotlib import pyplot as plt
import osmnx as ox
from shapely import MultiPolygon, Polygon

# Add imports for route merging
from src.routes.mutable_route import *

# Assuming these are available in your project structure
from src.routes.rating_system import RatingSystem
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
        self.established_routes = []  # Initialize established routes
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

    def get_processed_route(self, rp: RouteProcessor, route: Route, selected_index: int) -> ProcessedRoute:
        """Get processed route - works for both Route and EstablishedRoute objects"""
        route_dict = self.processed_routes
        
        if selected_index in route_dict:     
            processed_route = self.processed_routes[selected_index]
        else:                
            processed_route = rp.process_route(route)
            route_dict[selected_index] = processed_route
            
        return processed_route    

    def get_all_routes_for_processing(self) -> List[Route]:
        """Get all routes for processing, including both original and established routes"""
        all_routes = []
        
        # Add original routes
        all_routes.extend(self.routes)
        
        # Add established routes as unified routes
        for established_route in self.established_routes:
            unified_route = established_route.get_unified_route(min_confidence=0.5)
            all_routes.append(unified_route)            
            
        return all_routes
    
    @staticmethod
    def plot_merged_route_with_std(established_route, original_routes=None):
        """
        Plot merged route with standard deviation bands and original routes.
        Assumes established_route.std_devs contains per-point stats.
        """
        if not established_route.points:
            print("No points to plot")
            return
            
        # Extract coordinates
        lats = [p.lat for p in established_route.points]
        lons = [p.lon for p in established_route.points]
        
        # Extract standard deviations (handle missing data)
        if hasattr(established_route, 'std_devs') and established_route.std_devs:
            lat_stds = [s.get('lat_std', 0) for s in established_route.std_devs]
            lon_stds = [s.get('lon_std', 0) for s in established_route.std_devs]
        else:
            lat_stds = [0] * len(lats)
            lon_stds = [0] * len(lons)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot original routes (if provided) - semi-transparent gray
        if original_routes:
            for i, route in enumerate(original_routes.values()):
                orig_lats = [p.lat for p in route.points]
                orig_lons = [p.lon for p in route.points]
                alpha = 0.4
                if len(original_routes) == 1:
                    plt.plot(orig_lons, orig_lats, 'k--', alpha=alpha, linewidth=1, label=f'Original Route')
                else:
                    plt.plot(orig_lons, orig_lats, 'k--', alpha=alpha, linewidth=1, 
                            label=f'Original Route {i+1}')
        
        # Plot main merged route
        plt.plot(lons, lats, 'b-', linewidth=2.5, label='Merged Route', zorder=5)
        
        # Add uncertainty bands (3σ for 99.7% confidence)
        sigma_multiplier = 1.5
        plt.fill_between(
            lons, 
            [lat - sigma_multiplier * std for lat, std in zip(lats, lat_stds)],
            [lat + sigma_multiplier * std for lat, std in zip(lats, lat_stds)],
            color='red', alpha=0.2, label=f'{sigma_multiplier}σ Lat Uncertainty', zorder=4
        )
        
        # Styling
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        title_routes = len(original_routes) if original_routes else len(established_route.original_routes)
        plt.title(f'Merged Route: {established_route.name}\n'
                f'({len(established_route.points)} points, {title_routes} source routes)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
        
    def load_valid_routes_and_tracks(self):        
        print_step("SpotLoader", "Loading routes and tracks...")
        
        # Collect all (Route, List[Track]) pairs from all GPX files
        loaded_route_track_pairs = LocalGPXLoader.load_all_gpx()
        
        self.routes = [] # Reset to store all Route objects (including EstablishedRoute)
        self.tracks = [] # Reset to be a flattened list of all valid Track objects

        # First pass: load all valid routes and tracks
        valid_routes = []
        for route, associated_tracks in loaded_route_track_pairs:
            if route.is_valid_route(self.polygon):
                valid_routes.append(route)
                # Set the route reference for each track
                for track in associated_tracks:
                    track.route = route
                self.tracks.extend(associated_tracks)

        print_step("SpotLoader", f"Found {len(valid_routes)} valid routes for '{self.name}'")
        print_step("SpotLoader", f"Found {len(self.tracks)} tracks for '{self.name}'")

        # Second pass: merge similar routes into established routes
        if len(valid_routes) > 1:
            print_step("RouteMerging", "Clustering similar routes and merging groups...")
            
            config = RouteSimilarityConfig()
            merger = create_route_merger(config)
            merger.build_route_graph(valid_routes)
            established_routes = merger.merge_routes()
            self.plot_merged_route_with_std(established_routes[0], established_routes[0].original_routes)
            
            self.routes = [
                Route(
                    name=er.name,
                    points=tuple(er.points),
                    elevations=tuple(er.elevations),
                    descriptions=tuple(er.descriptions),
                    total_distance=er.total_distance
                )
                for er in established_routes
            ]            
            
        else:
            # No merging needed if only one route
            self.routes = valid_routes        
        
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