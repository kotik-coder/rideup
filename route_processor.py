import numpy as np
from scipy.interpolate import Akima1DInterpolator, interp1d
from typing import List, Dict, Tuple
from map_helpers import print_step
from route import GeoPoint, Route
from statistics_collector import StatisticsCollector, get_landscape_description
from checkpoints import CheckpointGenerator
from track import Track

PHOTO_CHECKPOINT_DISTANCE_THRESHOLD = 50.0  # meters

class RouteProcessor:
    def __init__(self, local_photos: List[Dict[str, any]], route_to_tracks: Dict[str, List[Track]]):
        self.checkpoint_generator = CheckpointGenerator(local_photos)
        self.route_to_tracks = route_to_tracks
        self.stats_collector = StatisticsCollector()

    def process_route(self, route: Route) -> Dict[str, any]:
        """Main processing pipeline for a route."""
        print_step("RouteProcessor", f"Starting processing for route: {route.name}")

        smooth_points, smooth_elevations = self._create_smooth_route(route)
        print_step("RouteProcessor", f"Route '{route.name}': Generated {len(smooth_points)} smoothed points.")

        associated_tracks = self.route_to_tracks.get(route.name, [])
        
        checkpoints = self.checkpoint_generator.generate_checkpoints(
            smooth_points, 
            smooth_elevations,
            associated_tracks
        )
        print_step("RouteProcessor", f"Route '{route.name}': Generated {len(checkpoints)} checkpoints.")
        
        segments = self._calculate_segments_and_enrich_descriptions(
            checkpoints, 
            smooth_elevations
        )
        print_step("RouteProcessor", f"Route '{route.name}': Generated {len(segments)} segments.")
        
        profiles = self.stats_collector.generate_route_profiles(
            route, 
            associated_tracks
        )

        return {
            'name': route.name,
            'checkpoints': checkpoints,
            'segments': segments,
            'elevation_profile': profiles['elevation_profile'],
            'velocity_profile': profiles['velocity_profile'],
            'raw_points': [(p.lat, p.lon) for p in route.points],
            'raw_elevations': route.elevations,
            'smooth_points': smooth_points
        }

    def _calculate_segments_and_enrich_descriptions(self, checkpoints: List[Dict[str, any]], 
                                                  smooth_elevations: List[float]) -> List[Dict[str, any]]:
        """Calculate route segments and enrich checkpoint descriptions with segment info."""
        segments = self._calculate_segments(checkpoints, smooth_elevations)
        
        for i in range(len(checkpoints)):
            cp = checkpoints[i]
            if i < len(segments):  # For start and intermediate points of segments
                segment_info = segments[i]
                cp['description'] = get_landscape_description(
                    current_elevation=cp['elevation'],
                    segment_net_elevation_change=segment_info['net_elevation'],
                    segment_elevation_gain=segment_info['elevation_gain'],
                    segment_elevation_loss=segment_info['elevation_loss'],
                    segment_distance=segment_info['distance']
                )
            elif i == len(checkpoints) - 1 and len(checkpoints) > 1:
                cp['description'] = "End of route."
            else:
                cp['description'] = "Start of route."
        
        return segments
    
    def _create_smooth_route(self, route: Route) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Creates a smoothed version of the route's points and elevations using
        linear or Akima spline interpolation based on point density.
        """
        MIN_POINTS_FOR_LINEAR = 50     # Use linear if ≥50 points
        MAX_DISTANCE_FOR_LINEAR = 10.0 # Use linear if avg spacing <10m
        
        if len(route.points) < 4:
            print_step("Smoothing", f"Route '{route.name}': <4 points, returning raw points")
            return [(p.lat, p.lon) for p in route.points], route.elevations

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
            return [(p.lat, p.lon) for p in route.points], route.elevations

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

            smooth_points = list(zip(
                interp_lat(t_smooth),
                interp_lon(t_smooth)
            ))
            return smooth_points, interp_elev(t_smooth).tolist()

        except Exception as e:
            print_step("Error", f"Smoothing failed for {route.name}: {str(e)}", level="ERROR")
            return [(p.lat, p.lon) for p in route.points], route.elevations

    def _calculate_segments(self, checkpoints: List[Dict[str, any]], elevations: List[float]) -> List[Dict[str, any]]:
        """
        Calculates segments between consecutive checkpoints, including distance and
        elevation changes within each segment.
        """
        segments = []
        if not checkpoints or len(checkpoints) < 2:
            print_step("RouteProcessor", "Segment calculation: No checkpoints or too few. Returning empty list.")
            return []

        for i in range(1, len(checkpoints)):
            start_cp = checkpoints[i-1]
            end_cp = checkpoints[i]

            start_idx = start_cp['point_index']
            end_idx = end_cp['point_index']

            segment_elevations = elevations[start_idx:end_idx+1]
            if not segment_elevations:
                print_step("RouteProcessor", f"Segment calculation: Segment from {start_idx} to {end_idx} has no elevation data. Skipping.", level="WARNING")
                continue

            min_seg_elevation = min(segment_elevations)
            max_seg_elevation = max(segment_elevations)
            
            segments.append({
                'distance': end_cp.get('distance_from_start', 0) - start_cp.get('distance_from_start', 0),
                'elevation_gain': max(0, max_seg_elevation - start_cp['elevation']),
                'elevation_loss': max(0, start_cp['elevation'] - min_seg_elevation),
                'net_elevation': end_cp['elevation'] - start_cp['elevation'],
                'min_segment_elevation': min_seg_elevation,
                'max_segment_elevation': max_seg_elevation,
                'start_checkpoint': start_cp['position'] - 1,
                'end_checkpoint': end_cp['position'] - 1
            })
        return segments