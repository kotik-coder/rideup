# route_processor.py
import numpy as np
from scipy.interpolate import Akima1DInterpolator, interp1d
from typing import List, Tuple

from src.ui.map_helpers import print_step
from src.routes.route import GeoPoint, Route
from src.routes.statistics_collector import StatisticsCollector, Segment
from src.routes.checkpoints import Checkpoint, CheckpointGenerator
from src.routes.track import Track
from src.iio.spot_photo import SpotPhoto

PHOTO_CHECKPOINT_DISTANCE_THRESHOLD = 50.0  # meters

class ProcessedRoute:
    smooth_points: List[GeoPoint]  # Interpolated points
    segments: List[Segment]
    checkpoints: List[Checkpoint]
    bounds: List[float]
    
    def __init__(self, smooth_points : List[GeoPoint]):
        self.smooth_points = smooth_points            
    
    def find_closest_route_point(self, point: GeoPoint) -> Tuple[int, GeoPoint]:

        index = min(
            range(len(self.smooth_points)),
            key=lambda i: self.smooth_points[i].distance_to(point)
        )
        
        return index, self.smooth_points[index]        

class RouteProcessor:
    def __init__(self, local_photos: List[SpotPhoto], all_tracks: List[Track]):
        self.checkpoint_generator = CheckpointGenerator(local_photos)
        self.all_tracks = all_tracks
        self.stats_collector = StatisticsCollector()

    def process_route(self, route: Route) -> ProcessedRoute:
        """Main processing pipeline for a route."""
    
        if not route:
            return        
        
        print_step("RouteProcessor", f"Starting processing for route: {route.name}")        

        smooth_points = self._create_smooth_route(route)
        print_step("RouteProcessor", f"Route '{route.name}': Generated {len(smooth_points)} smoothed points.")

        # Get tracks associated with this route
        associated_tracks = [t for t in self.all_tracks if t.route == route]        
        processed_route = ProcessedRoute( smooth_points )
        
        processed_route.checkpoints = self.checkpoint_generator.generate_checkpoints(
            processed_route.smooth_points,
            associated_tracks
        )
        print_step("RouteProcessor", f"Route '{route.name}': Generated {len(processed_route.checkpoints)} checkpoints.")
        
        processed_route.segments = self._calculate_segments(processed_route)
        
        lons = [p.lon for p in route.points]
        lats = [p.lat for p in route.points]
        
        processed_route.bounds = [min(lons), min(lats), max(lons), max(lats)]            

        print_step("RouteProcessor", f"Route '{route.name}': Generated {len(processed_route.segments)} segments.")

        return processed_route
    
    def _calculate_segments(self, route : ProcessedRoute) -> List[Segment]:
        """Calculates segments between consecutive checkpoints."""
        checkpoints = route.checkpoints
        if len(checkpoints) < 2:
            print_step("RouteProcessor", "Segment calculation: Not enough checkpoints. Returning empty list.")
            return []

        segments = []
        for i in range(1, len(checkpoints)):
            start_cp = checkpoints[i-1]
            end_cp   = checkpoints[i]

            start_idx = start_cp.route_point_index
            end_idx   = end_cp.route_point_index
            
            route_point_start = route.smooth_points[start_idx]
            route_point_end   = route.smooth_points[end_idx]                        
            
            segment_elevations = [rp.elevation for rp in route.smooth_points[start_idx:end_idx+1]]
            if not segment_elevations:
                print_step("RouteProcessor", f"Segment calculation: Segment from {start_idx} to {end_idx} has no elevation data. Skipping.", level="WARNING")
                continue

            min_seg_elevation = min(segment_elevations)
            max_seg_elevation = max(segment_elevations)
            
            segments.append(Segment(
                distance=route_point_start.distance_to(route_point_end),
                elevation_gain=max(0, max_seg_elevation - route_point_start.elevation),
                elevation_loss=max(0, route_point_start.elevation - min_seg_elevation),
                net_elevation=route_point_end.elevation - route_point_start.elevation,
                min_elevation=min_seg_elevation,
                max_elevation=max_seg_elevation,
                start_checkpoint_index=i-1,
                end_checkpoint_index=i
            ))
        return segments

    def _create_smooth_route(self, route: Route) -> List[GeoPoint]:
        """
        Creates a smoothed version of the route's points and elevations using
        linear or Akima spline interpolation based on point density.
        """
        MIN_POINTS_FOR_LINEAR = 100    # Use linear if ≥ 200 points
        MAX_DISTANCE_FOR_LINEAR = 15.0 # Use linear if avg spacing <15m

        points     = route.points

        distances = np.zeros(len(points))
        for i in range(1, len(points)):
            distances[i] = distances[i-1] + points[i-1].distance_to(points[i])        
                
        avg_distance = distances[-1] / (len(points) - 1)
        
        use_linear = (len(points) >= MIN_POINTS_FOR_LINEAR or 
                     avg_distance <= MAX_DISTANCE_FOR_LINEAR)
        
        method = "linear" if use_linear else "Akima"
        print_step("Smoothing", 
                f"Route '{route.name}': {len(points)} points, "
                f"avg spacing {avg_distance:.1f}m → {method} interpolation")

        t = distances / distances[-1]
        num_smooth_points = int(max(MIN_POINTS_FOR_LINEAR, 
                                distances[-1] / MAX_DISTANCE_FOR_LINEAR))
        t_smooth = np.linspace(0, 1, num_smooth_points)

        try:
            
            lats = [p.lat for p in points]
            lons = [p.lon for p in points]
            
            if use_linear:
                interp_lat = interp1d(t,  lats, kind='linear')
                interp_lon = interp1d(t,  lons, kind='linear')
                interp_elev = interp1d(t, route.elevations, kind='linear')
            else:
                interp_lat = Akima1DInterpolator(t, lats)
                interp_lon = Akima1DInterpolator(t, lons)
                interp_elev = Akima1DInterpolator(t, route.elevations)

            smooth_points = [
                GeoPoint(lat, lon, elev) 
                for lat, lon, elev in zip(
                    interp_lat(t_smooth),
                    interp_lon(t_smooth),
                    interp_elev(t_smooth)
                )
            ]
            return smooth_points

        except Exception as e:
            print_step("Error", f"Smoothing failed for {route.name}: {str(e)}", level="ERROR")
            return route.points.copy()