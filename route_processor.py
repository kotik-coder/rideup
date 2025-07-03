import numpy as np
from scipy.interpolate import Akima1DInterpolator, interp1d
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional

# Assuming these are available in your project structure
from map_helpers import print_step
from route import Route, GeoPoint
from track import Track, TrackPoint
from media_helpers import get_photo_html # Assuming this is available
from statistics_collector import StatisticsCollector, get_landscape_description

PHOTO_CHECKPOINT_DISTANCE_THRESHOLD = 50.0 # meters

class RouteProcessor:
    """
    Handles the detailed processing of individual routes, including smoothing,
    generating checkpoints, and calculating segments.
    """
    def __init__(self, local_photos: List[Dict[str, any]], route_to_tracks: Dict[str, List[Track]]):
        """
        Initializes the RouteProcessor with data needed for checkpoint generation
        and track association.

        Args:
            local_photos (List[Dict[str, any]]): List of local photos with metadata.
            route_to_tracks (Dict[str, List[Track]]): Mapping from route names to their tracks.
        """
        self.local_photos = local_photos
        self.route_to_tracks = route_to_tracks

    def process_route(self, route: Route) -> Dict[str, any]:
        """Processes a single Route object to generate comprehensive data."""
        print_step("RouteProcessor", f"Starting processing for route: {route.name}")

        smooth_points, smooth_elevations = self._create_smooth_route(route)
        print_step("RouteProcessor", f"Route '{route.name}': Generated {len(smooth_points)} smoothed points.")

        associated_tracks = self.route_to_tracks.get(route.name, [])

        # Generate profiles using StatisticsCollector
        stats_collector = StatisticsCollector()
        profiles = stats_collector.generate_route_profiles(route, associated_tracks)

        # Calculate distances for checkpoints first
        distances = [0.0]
        for i in range(1, len(smooth_points)):
            p1 = GeoPoint(*smooth_points[i-1])
            p2 = GeoPoint(*smooth_points[i])
            distances.append(distances[-1] + p1.distance_to(p2))

        # Generate checkpoints (without full descriptions initially)
        checkpoints = self._get_checkpoints(route, smooth_points, smooth_elevations, 
                                          self.local_photos, associated_tracks)
        print_step("RouteProcessor", f"Route '{route.name}': Generated {len(checkpoints)} checkpoints.")

        # Assign distances to checkpoints
        if smooth_points and checkpoints:
            total_dist_so_far = 0
            current_smooth_idx = 0
            for cp in checkpoints:
                target_smooth_idx = cp['point_index']
                # Accumulate distance from the last processed smooth point index to the current checkpoint's index
                for k in range(current_smooth_idx, min(target_smooth_idx, len(smooth_points) - 1)):
                    p1 = GeoPoint(*smooth_points[k])
                    p2 = GeoPoint(*smooth_points[k+1])
                    total_dist_so_far += p1.distance_to(p2)
                cp['distance_from_start'] = total_dist_so_far
                current_smooth_idx = target_smooth_idx
        else:
            print_step("RouteProcessor", f"Route '{route.name}': Skipped checkpoint distance calculation due to missing smooth points or checkpoints.")

        # Now, calculate segments and then enrich checkpoint descriptions
        segments = self._calculate_segments(checkpoints, smooth_elevations)
        print_step("RouteProcessor", f"Route '{route.name}': Generated {len(segments)} segments.")
        
        # Enrich checkpoint descriptions using segment information
        # Iterate through checkpoints (excluding the very last one, as it's an end point of a segment)
        for i in range(len(checkpoints)):
            cp = checkpoints[i]
            if i < len(segments): # For start and intermediate points of segments
                segment_info = segments[i]
                # Pass current checkpoint elevation and segment stats to get_landscape_description
                # Assuming get_landscape_description is available (e.g., in map_helpers or media_helpers)
                cp['description'] = get_landscape_description( # Using a placeholder for now
                    current_elevation=cp['elevation'],
                    segment_net_elevation_change=segment_info['net_elevation'],
                    segment_elevation_gain=segment_info['elevation_gain'],
                    segment_elevation_loss=segment_info['elevation_loss'],
                    segment_distance=segment_info['distance']
                )
            elif i == len(checkpoints) - 1 and len(checkpoints) > 1:
                # This is the very last checkpoint (the finish point)
                cp['description'] = "End of route."
            else: # For the very first checkpoint, set a general start description
                cp['description'] = "Start of route."

        # Velocity profile is now handled by StatisticsCollector, but RouteProcessor
        # might need to trigger its creation and include it in the processed_route_dict.
        # For now, we'll keep the logic of getting it from associated_tracks[0]
        # if it's still intended to be part of the RouteProcessor's output.
        velocity_profile = []
        if associated_tracks:
            # Assuming Track objects have _create_velocity_profile method
            velocity_profile = StatisticsCollector().create_velocity_profile_from_track(associated_tracks[0])
            print_step("RouteProcessor", f"Route '{route.name}': Generated velocity profile with {len(velocity_profile)} points.")

        processed_route_dict = {
            'name': route.name,
            'checkpoints': checkpoints,
            'segments': segments,
            'elevation_profile': profiles['elevation_profile'],  # From StatisticsCollector
            'velocity_profile': profiles['velocity_profile'],    # From StatisticsCollector
            'raw_points': [(p.lat, p.lon) for p in route.points],
            'raw_elevations': route.elevations,
            'smooth_points': smooth_points
        }

        print_step("RouteProcessor", f"Finished processing for route: {route.name}.")
        return processed_route_dict

    def _create_smooth_route(self, route: Route) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Creates a smoothed version of the route's points and elevations using
        linear or Akima spline interpolation based on point density.
        """
        # Thresholds for interpolation method selection
        MIN_POINTS_FOR_LINEAR = 50     # Use linear if ≥50 points
        MAX_DISTANCE_FOR_LINEAR = 10.0 # Use linear if avg spacing <10m
        
        if len(route.points) < 4:
            print_step("Smoothing", f"Route '{route.name}': <4 points, returning raw points")
            return [(p.lat, p.lon) for p in route.points], route.elevations

        points = np.array([(p.lat, p.lon) for p in route.points])
        elevations = np.array(route.elevations)

        # Calculate cumulative distances and average spacing
        distances = np.zeros(len(points))
        for i in range(1, len(points)):
            p1 = GeoPoint(points[i-1][0], points[i-1][1])
            p2 = GeoPoint(points[i][0], points[i][1])
            distances[i] = distances[i-1] + p1.distance_to(p2)
        
        avg_distance = distances[-1] / (len(points) - 1) if len(points) > 1 else 0
        
        # Determine interpolation method
        use_linear = (len(points) >= MIN_POINTS_FOR_LINEAR and 
                    avg_distance <= MAX_DISTANCE_FOR_LINEAR)
        
        method = "linear" if use_linear else "Akima"
        print_step("Smoothing", 
                f"Route '{route.name}': {len(points)} points, "
                f"avg spacing {avg_distance:.1f}m → {method} interpolation")

        if distances[-1] == 0:
            return [(p.lat, p.lon) for p in route.points], route.elevations

        # Normalized distance parameter (0-1)
        t = distances / distances[-1]
        num_smooth_points = max(100, int(distances[-1] / 10))
        t_smooth = np.linspace(0, 1, num_smooth_points)

        try:
            if use_linear:
                # Linear interpolation - preserves sharp turns in dense data
                interp_lat  = interp1d(t, points[:, 0], kind='linear')
                interp_lon  = interp1d(t, points[:, 1], kind='linear')
                interp_elev = interp1d(t, elevations, kind='linear')
            else:
                # Akima spline - smooths sparse data
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

    def _get_checkpoints(self, route: Route, smooth_route: List[Tuple[float, float]], 
                        route_elevations: List[float], local_photos: List[Dict[str, any]],
                        associated_tracks: List[Track]) -> List[Dict[str, any]]:
        """
        Generates checkpoints along the smooth route, including start/end,
        uniformly spaced points, and points near local photos.
        """
        if len(smooth_route) < 2:
            print_step("RouteProcessor", "Checkpoint generation: Too few smooth points. Returning empty list.")
            return []

        # Calculate cumulative distances along the smooth route
        distances = [0.0]
        for i in range(1, len(smooth_route)):
            p1 = GeoPoint(*smooth_route[i-1])
            p2 = GeoPoint(*smooth_route[i])
            distances.append(distances[-1] + p1.distance_to(p2))

        total_length = distances[-1]

        marker_indices = set()
        index_to_photo_info: Dict[int, Dict[str, any]] = {} 

        # Always add start and end points
        marker_indices.add(0)
        marker_indices.add(len(smooth_route) - 1)

        # Add checkpoints based on local photos near associated tracks
        for track in associated_tracks:
            if not track.points:
                continue

            track_start_time = track.points[0].timestamp if track.points else None
            track_end_time = track.points[-1].timestamp if track.points else None

            if track_start_time and track_end_time:
                track_duration = (track_end_time - track_start_time).total_seconds()
                
                # Sort photos by timestamp to optimize search
                local_photos.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min.replace(tzinfo=timezone.utc))

                for photo in local_photos:
                    # Skip if this photo has already been used for a checkpoint
                    if photo['path'] in [info.get('path') for info in index_to_photo_info.values()]:
                        continue 

                    photo_time = photo.get('timestamp')
                    if photo_time:
                        photo_elapsed = (photo_time - track_start_time).total_seconds()
                        
                        # Check if photo timestamp is within track's duration
                        if 0 <= photo_elapsed <= track_duration:
                            closest_track_point: Optional[TrackPoint] = None
                            min_time_diff = float('inf')
                            
                            # Find the closest track point by elapsed time
                            for tp in track.points:
                                time_diff = abs(photo_elapsed - tp.elapsed_seconds)
                                if time_diff < min_time_diff:
                                    min_time_diff = time_diff
                                    closest_track_point = tp
                            
                            # If a sufficiently close track point is found
                            if closest_track_point is not None and min_time_diff < 300: # 5 minutes threshold
                                # Find the closest smooth route point to this track point's geographic coordinates
                                closest_smooth_idx = min(
                                    range(len(smooth_route)),
                                    key=lambda x: GeoPoint(*smooth_route[x]).distance_to(closest_track_point.point)
                                )
                                marker_indices.add(closest_smooth_idx)
                                index_to_photo_info[closest_smooth_idx] = photo

        # Add uniformly spaced markers
        target_markers_uniform = min(20, max(5, int(total_length / 250))) if total_length > 0 else 2
        if target_markers_uniform > 2 and total_length > 0:
            step_length = total_length / (target_markers_uniform - 1)
            for i in range(1, target_markers_uniform - 1):
                target_dist = i * step_length
                closest_idx = min(range(len(distances)), key=lambda x: abs(distances[x] - target_dist))
                marker_indices.add(closest_idx)

        sorted_indices = sorted(list(marker_indices))
        
        checkpoints = []

        for i, idx in enumerate(sorted_indices):
            point = smooth_route[idx]
            elevation = route_elevations[idx]
            
            photo_info = index_to_photo_info.get(idx)

            point_name = f"Точка {i+1}"
            description = "" # Description will be set later by process_route
            photo_html = ""

            if photo_info:
                point_name = f"Фототочка"
                photo_html = get_photo_html(point[0], point[1], local_photo_path=photo_info['path'])
            else:
                photo_html = get_photo_html(point[0], point[1])

            if i == 0:
                point_name = "Старт"
                description = "Старт маршрута." # Initial description for start
            elif i == len(sorted_indices) - 1:
                point_name = "Финиш"
                description = "Конец маршрута." # Initial description for finish


            checkpoint = {
                'point_index': idx,
                'position': i + 1,
                'total_positions': len(sorted_indices),
                'lat': point[0],
                'lon': point[1],
                'elevation': elevation,
                'name': point_name,
                'description': description, # Will be overwritten later for intermediate points
                'photo_html': photo_html
            }
            checkpoints.append(checkpoint)

        return checkpoints

    def _calculate_segments(self, checkpoints: List[dict], elevations: List[float]) -> List[Dict[str, any]]:
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

            # Ensure end_idx is inclusive for segment elevations
            segment_elevations = elevations[start_idx:end_idx+1]
            if not segment_elevations:
                print_step("RouteProcessor", f"Segment calculation: Segment from {start_idx} to {end_idx} has no elevation data. Skipping.", level="WARNING")
                continue

            # Calculate min and max elevation within the segment
            min_seg_elevation = min(segment_elevations) if segment_elevations else 0.0
            max_seg_elevation = max(segment_elevations) if segment_elevations else 0.0
            
            segments.append({
                'distance': end_cp.get('distance_from_start', 0) - start_cp.get('distance_from_start', 0),
                'elevation_gain': max(0, max_seg_elevation - start_cp['elevation']), # Net gain from start_cp
                'elevation_loss': max(0, start_cp['elevation'] - min_seg_elevation), # Net loss from start_cp
                'net_elevation': end_cp['elevation'] - start_cp['elevation'],
                'min_segment_elevation': min_seg_elevation, # Min elevation for the segment
                'max_segment_elevation': max_seg_elevation, # Max elevation for the segment
                'start_checkpoint': start_cp['position'] - 1,
                'end_checkpoint': end_cp['position'] - 1
            })
        return segments