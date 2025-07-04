import osmnx as ox
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

# Assuming these are available in your project structure
from map_helpers import print_step
import gpx_loader
from gpx_loader import LocalGPXLoader # Assuming LocalGPXLoader is within gpx_loader
from route import Route
from spot_photo import SpotPhoto
from track import Track
from media_helpers import get_exif_geolocation, get_photo_timestamp

@dataclass
class Spot:
    name: str
    geostring: str
    bounds: List[float]
    local_photos_folder: str
    local_photos: List[SpotPhoto]
    routes: List[Route]
    tracks: List[Track]  # Tracks now contain their route reference

    def __init__(self, geostring: str, local_photos_folder: str = "local_photos"):
        self.name = geostring
        self.geostring = geostring
        self.local_photos_folder = local_photos_folder
        self.bounds = self._get_forest_bounds(geostring)
        self.local_photos = []
        self.routes = []
        self.tracks = []
        print_step("Spot", f"Spot '{self.name}' initialized with bounds: {self.bounds}")

    def _get_forest_bounds(self, geostring: str) -> List[float]:
        """
        Retrieves the geographical bounds of the forest using OSMnx.
        Returns a list [min_lon, min_lat, max_lon, max_lat].
        """
        print_step(prefix="Spot", message=f"Getting bounds for '{geostring}'...")
        try:
            forest = ox.geocode_to_gdf(geostring)
            bounds = [float(x) for x in forest.total_bounds]
            print_step("Spot", f"Bounds obtained: {bounds}")
            return bounds
        except Exception as e:
            print_step("Error", f"Failed to get bounds for '{geostring}': {e}", level="ERROR")
            # Return a sensible default or raise an error if bounds are critical
            return [37.5, 55.5, 37.6, 55.6] # Example default bounds for Moscow area

class SpotLoader:
    """
    Responsible for loading and populating a Spot object with valid routes,
    tracks, and associated local photos.
    """
    def __init__(self, spot: Spot):
        self.spot = spot
        self._photos_loaded = False # Internal flag to prevent redundant photo loading

    def _load_local_photos(self) -> List[SpotPhoto]:
        """
        Loads local photos with geolocation and timestamp data from the specified folder
        associated with the spot.
        """
        if self._photos_loaded:
            return self.spot.local_photos

        folder_path_str = self.spot.local_photos_folder
        print_step("SpotLoader", f"Loading local photos from folder: {folder_path_str}...")
        folder_path = Path(folder_path_str)
        if not folder_path.is_dir():
            print_step("SpotLoader", f"Folder '{folder_path_str}' not found or is not a directory.", level="WARNING")
            return []
        
        photos = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for image_path in folder_path.glob(ext):
                photo = SpotPhoto.from_image_path(image_path)
                if photo:
                    photos.append(photo)
        
        print_step("SpotLoader", f"Found {len(photos)} photos with timestamps.")
        self.spot.local_photos = photos
        self._photos_loaded = True
        return self.spot.local_photos

    def load_valid_routes_and_tracks(self):
        """
        Loads routes and tracks from GPX files, filters them based on the spot's bounds,
        and populates the Spot object.
        """
        print_step("SpotLoader", "Loading routes and tracks...")
        loader = LocalGPXLoader()
        
        # Collect all (Route, List[Track]) pairs from all GPX files
        loaded_route_track_pairs: List[Tuple[Route, List[Track]]] = []
        
        for gpx_file in gpx_loader.GPX_DIR.glob("*.gpx"):                        
            loaded_route_track_pairs.extend(loader.load_routes_and_tracks(gpx_file))

        self.spot.routes = [] # Reset to store only valid Route objects
        self.spot.tracks = [] # Reset to be a flattened list of all valid Track objects

        for route, associated_tracks in loaded_route_track_pairs:
            if route.is_valid_route(self.spot.bounds):
                self.spot.routes.append(route)
                # Set the route reference for each track
                for track in associated_tracks:
                    track.route = route
                self.spot.tracks.extend(associated_tracks)
            else:
                print_step("SpotLoader", f"Route '{route.name}' is outside spot bounds or invalid. Skipping.", level="WARN")

        print_step("SpotLoader", f"Found {len(self.spot.routes)} valid routes for '{self.spot.name}'")
        print_step("SpotLoader", f"Found {len(self.spot.tracks)} tracks for '{self.spot.name}'")
        
        # Ensure local photos are loaded after routes and tracks, as they might be used
        # for checkpoint generation by RouteProcessor.
        self._load_local_photos()

        return self.spot.routes