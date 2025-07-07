# route_processor.py
import numpy as np
from scipy.interpolate import Akima1DInterpolator, interp1d
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from map_helpers import print_step
from route import GeoPoint, Route
from spot import Spot
from statistics_collector import StatisticsCollector, get_landscape_description, Segment
from checkpoints import Checkpoint, CheckpointGenerator
from track import Track
from spot_photo import SpotPhoto

PHOTO_CHECKPOINT_DISTANCE_THRESHOLD = 50.0  # meters

@dataclass
class ProcessedRoute:
    """Contains the original route and processed data"""
    route: Route
    smooth_points: List[GeoPoint]  # Interpolated points
    segments: List[Segment]
    checkpoints: List[Checkpoint]    

class RouteProcessor:
    def __init__(self, local_photos: List[SpotPhoto], all_tracks: List[Track]):
        self.checkpoint_generator = CheckpointGenerator(local_photos)
        self.all_tracks = all_tracks
        self.stats_collector = StatisticsCollector()

    def process_route(self, route: Route) -> ProcessedRoute:
        """Main processing pipeline for a route."""
        print_step("RouteProcessor", f"Starting processing for route: {route.name}")

        smooth_points, smooth_elevations = self._create_smooth_route(route)
        print_step("RouteProcessor", f"Route '{route.name}': Generated {len(smooth_points)} smoothed points.")

        # Get tracks associated with this route
        associated_tracks = [t for t in self.all_tracks if t.route == route]
        
        checkpoints = self.checkpoint_generator.generate_checkpoints(
            [(p.lat, p.lon) for p in smooth_points],
            smooth_elevations,
            associated_tracks
        )
        print_step("RouteProcessor", f"Route '{route.name}': Generated {len(checkpoints)} checkpoints.")
        
        segments = self._calculate_segments(checkpoints, smooth_elevations)
        self._enrich_descriptions(checkpoints, segments)

        print_step("RouteProcessor", f"Route '{route.name}': Generated {len(segments)} segments.")

        return ProcessedRoute(
            route=route,
            smooth_points=smooth_points,
            segments=segments,
            checkpoints=checkpoints
        )

    def _enrich_descriptions(self, checkpoints: List[Checkpoint], segments: List[Segment]):
        for i, checkpoint in enumerate(checkpoints):
            if i < len(segments):
                segment = segments[i]
                checkpoint.description = get_landscape_description(
                    current_elevation=checkpoint.elevation,
                    segment_net_elevation_change=segment.net_elevation,
                    segment_elevation_gain=segment.elevation_gain,
                    segment_elevation_loss=segment.elevation_loss,
                    segment_distance=segment.distance
                )
            elif i == len(checkpoints) - 1:
                checkpoint.description = "End of route."
            else:
                checkpoint.description = "Start of route."
        
        return segments
    
    def _calculate_segments(self, checkpoints: List[Checkpoint], elevations: List[float]) -> List[Segment]:
        """Calculates segments between consecutive checkpoints."""
        if len(checkpoints) < 2:
            print_step("RouteProcessor", "Segment calculation: Not enough checkpoints. Returning empty list.")
            return []

        segments = []
        for i in range(1, len(checkpoints)):
            start_cp = checkpoints[i-1]
            end_cp = checkpoints[i]

            start_idx = start_cp.point_index
            end_idx = end_cp.point_index

            segment_elevations = elevations[start_idx:end_idx+1]
            if not segment_elevations:
                print_step("RouteProcessor", f"Segment calculation: Segment from {start_idx} to {end_idx} has no elevation data. Skipping.", level="WARNING")
                continue

            min_seg_elevation = min(segment_elevations)
            max_seg_elevation = max(segment_elevations)
            
            segments.append(Segment(
                distance=end_cp.distance_from_start - start_cp.distance_from_start,
                elevation_gain=max(0, max_seg_elevation - start_cp.elevation),
                elevation_loss=max(0, start_cp.elevation - min_seg_elevation),
                net_elevation=end_cp.elevation - start_cp.elevation,
                min_elevation=min_seg_elevation,
                max_elevation=max_seg_elevation,
                start_checkpoint_index=start_cp.position - 1,
                end_checkpoint_index=end_cp.position - 1
            ))
        return segments

    def _create_smooth_route(self, route: Route) -> Tuple[List[GeoPoint], List[float]]:
        """
        Creates a smoothed version of the route's points and elevations using
        linear or Akima spline interpolation based on point density.
        """
        MIN_POINTS_FOR_LINEAR = 50     # Use linear if ≥50 points
        MAX_DISTANCE_FOR_LINEAR = 10.0 # Use linear if avg spacing <10m
        
        if len(route.points) < 4:
            print_step("Smoothing", f"Route '{route.name}': <4 points, returning raw points")
            return route.points.copy(), route.elevations.copy()

        points = np.array([(p.lat, p.lon) for p in route.points])
        elevations = np.array(route.elevations)

        distances = np.zeros(len(points))
        for i in range(1, len(points)):
            p1 = GeoPoint(points[i-1][0], points[i-1][1])
            p2 = GeoPoint(points[i][0], points[i][1])
            distances[i] = distances[i-1] + p1.distance_to(p2)
        
        avg_distance = distances[-1] / (len(points) - 1) if len(points) > 1 else 0
        
        use_linear = (len(points) >= MIN_POINTS_FOR_LINEAR and 
                    avg_distance <= MAX_DISTANCE_FOR_LINEAR)
        
        method = "linear" if use_linear else "Akima"
        print_step("Smoothing", 
                f"Route '{route.name}': {len(points)} points, "
                f"avg spacing {avg_distance:.1f}m → {method} interpolation")

        if distances[-1] == 0:
            return route.points.copy(), route.elevations.copy()

        t = distances / distances[-1]
        num_smooth_points = max(100, int(distances[-1] / 10))
        t_smooth = np.linspace(0, 1, num_smooth_points)

        try:
            if use_linear:
                interp_lat = interp1d(t, points[:, 0], kind='linear')
                interp_lon = interp1d(t, points[:, 1], kind='linear')
                interp_elev = interp1d(t, elevations, kind='linear')
            else:
                interp_lat = Akima1DInterpolator(t, points[:, 0])
                interp_lon = Akima1DInterpolator(t, points[:, 1])
                interp_elev = Akima1DInterpolator(t, elevations)

            smooth_points = [
                GeoPoint(lat, lon) 
                for lat, lon in zip(
                    interp_lat(t_smooth),
                    interp_lon(t_smooth)
                )
            ]
            return smooth_points, interp_elev(t_smooth).tolist()

        except Exception as e:
            print_step("Error", f"Smoothing failed for {route.name}: {str(e)}", level="ERROR")
            return route.points.copy(), route.elevations.copy()