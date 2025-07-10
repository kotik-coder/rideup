import os
import gpxpy
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from routes.route import Route, GeoPoint
from ui.map_helpers import print_step
from routes.track import Track, TrackPoint

# Get the package root directory
package_root = Path(__file__).parent.parent
GPX_DIR      = package_root / "local_routes"

class LocalGPXLoader:
    def __init__(self):
        print_step("GPX", "Инициализация загрузчика локальных маршрутов")
        GPX_DIR.mkdir(exist_ok=True)

    def _extract_metadata(self, gpx) -> Dict:
        """Извлекает метаданные из GPX файла"""
        metadata = {
            "name": "Безымянный маршрут",
            "desc": "",
            "time": None
        }
        
        if gpx.name:
            metadata["name"] = gpx.name
        elif gpx.tracks and gpx.tracks[0].name:
            metadata["name"] = gpx.tracks[0].name
            
        if gpx.description:
            metadata["desc"] = gpx.description
        elif gpx.tracks and gpx.tracks[0].description:
            metadata["desc"] = gpx.tracks[0].description
            
        if gpx.time:
            if gpx.time.tzinfo is None:
                metadata["time"] = gpx.time.replace(tzinfo=timezone.utc)
            else:
                metadata["time"] = gpx.time
        return metadata

    def load_routes_and_tracks(self, gpx_path: Path) -> List[Tuple[Route, List[Track]]]:
        results: List[Tuple[Route, List[Track]]] = []
        with open(gpx_path, 'r') as gpx_file:
            gpx = gpxpy.parse(gpx_file)

        for track_gpx in gpx.tracks:
            current_route_points = []
            current_elevations = []
            current_descriptions = []
            current_tracks_for_route: List[Track] = []
            total_distance = 0.0  # Initialize total distance

            for segment in track_gpx.segments:
                # Collect data for the Route object
                if segment.points:
                    # Calculate distance between points for the route
                    prev_point = None
                    for pt in segment.points:
                        geo_point = GeoPoint(pt.latitude, pt.longitude, pt.elevation or 0)
                        current_route_points.append(geo_point)
                        current_elevations.append(pt.elevation or 0)
                        current_descriptions.append(pt.description or "")
                        
                        if prev_point:
                            total_distance += prev_point.distance_to(geo_point)
                        prev_point = geo_point
                    
                # Create Track object from segment points
                track_points_for_segment = []
                start_time = self._ensure_utc(segment.points[0].time) if segment.points and segment.points[0].time else None
                
                for pt in segment.points:
                    timestamp = self._ensure_utc(pt.time)
                    elapsed = (timestamp - start_time).total_seconds() if start_time and timestamp else 0
                    track_points_for_segment.append(TrackPoint(
                        point=GeoPoint(pt.latitude, pt.longitude, pt.elevation or 0),
                        timestamp=timestamp,
                        elapsed_seconds=elapsed
                    ))
                
                if track_points_for_segment:
                    current_tracks_for_route.append(Track(track_points_for_segment))
            
            if current_route_points:
                route_obj = Route(
                    name=track_gpx.name or gpx_path.stem,
                    points=current_route_points,
                    elevations=current_elevations,
                    descriptions=current_descriptions,
                    total_distance=total_distance  # Set the calculated total distance
                )
                
                # Set route reference for all tracks
                for track in current_tracks_for_route:
                    track.route = route_obj
                
                results.append((route_obj, current_tracks_for_route))

        return results

    def _ensure_utc(self, dt: Optional[datetime]) -> Optional[datetime]:
        if dt and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt