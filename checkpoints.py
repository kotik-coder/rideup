from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime, timezone
from route import GeoPoint
from track import Track, TrackPoint
from media_helpers import get_photo_html

PHOTO_CHECKPOINT_DISTANCE_THRESHOLD = 50.0  # meters

class CheckpointGenerator:
    def __init__(self, local_photos: List[Dict[str, any]]):
        self.local_photos = local_photos
        self.index_to_photo_info: Dict[int, Dict[str, any]] = {}

    def generate_checkpoints(self, smooth_route: List[Tuple[float, float]], 
                        route_elevations: List[float],
                        associated_tracks: List[Track]) -> List[Dict[str, any]]:
        """Main method to generate all checkpoints for a route."""
        if len(smooth_route) < 2:
            return []

        distances = self._calculate_smooth_route_distances(smooth_route)
        marker_indices = self._collect_marker_indices(
            smooth_route, 
            distances[-1], 
            distances,
            associated_tracks
        )
        return self._create_checkpoints_from_markers(
            smooth_route, 
            route_elevations, 
            marker_indices
        )

    def _calculate_smooth_route_distances(self, smooth_route: List[Tuple[float, float]]) -> List[float]:
        """Calculate cumulative distances along the smooth route."""
        distances = [0.0]
        for i in range(1, len(smooth_route)):
            p1 = GeoPoint(*smooth_route[i-1])
            p2 = GeoPoint(*smooth_route[i])
            distances.append(distances[-1] + p1.distance_to(p2))
        return distances

    def _collect_marker_indices(self, smooth_route: List[Tuple[float, float]], 
                              total_length: float,
                              distances: List[float],
                              associated_tracks: List[Track]) -> List[int]:
        """Collect all marker indices including start/end, photo points, and uniform points."""
        marker_indices = set()
        marker_indices.add(0)
        marker_indices.add(len(smooth_route) - 1)
        
        self._add_photo_markers(marker_indices, smooth_route, associated_tracks)
        self._add_uniform_markers(marker_indices, distances, total_length)
        
        return sorted(marker_indices)

    def _add_photo_markers(self, marker_indices: Set[int], 
                         smooth_route: List[Tuple[float, float]], 
                         associated_tracks: List[Track]) -> None:
        """Add markers for points near local photos."""
        for track in associated_tracks:
            if not track.points:
                continue

            track_start_time = track.points[0].timestamp
            track_end_time = track.points[-1].timestamp
            track_duration = (track_end_time - track_start_time).total_seconds()

            for photo in sorted(self.local_photos, key=lambda x: x.get('timestamp', datetime.min.replace(tzinfo=timezone.utc))):
                if photo['path'] in [p.get('path') for p in self.index_to_photo_info.values()]:
                    continue
                
                self._process_photo_for_marker(
                    photo, track, track_start_time, 
                    track_duration, smooth_route, 
                    marker_indices
                )

    def _process_photo_for_marker(self, photo: Dict[str, any], track: Track, 
                            track_start_time: datetime, track_duration: float,
                            smooth_route: List[Tuple[float, float]], 
                            marker_indices: Set[int]) -> None:
        """Process a single photo to potentially add a marker."""
        photo_time = photo.get('timestamp')
        if not photo_time:
            return

        photo_elapsed = (photo_time - track_start_time).total_seconds()
        if not (0 <= photo_elapsed <= track_duration):
            return

        closest_track_point = self._find_closest_track_point(track.points, photo_elapsed)
        if closest_track_point is None or abs(photo_elapsed - closest_track_point.elapsed_seconds) >= 300:
            return

        closest_smooth_idx = self._find_closest_smooth_point(smooth_route, closest_track_point.point)
        marker_indices.add(closest_smooth_idx)
        # Store the full photo info in the dictionary
        self.index_to_photo_info[closest_smooth_idx] = {
            'path': photo['path'],
            'timestamp': photo_time,
            'coords': photo.get('coords')
        }
        
    

    def _find_closest_track_point(self, track_points: List[TrackPoint], photo_elapsed: float) -> Optional[TrackPoint]:
        """Find the track point closest to the photo's timestamp."""
        closest_track_point = None
        min_time_diff = float('inf')
        
        for tp in track_points:
            time_diff = abs(photo_elapsed - tp.elapsed_seconds)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_track_point = tp
        
        return closest_track_point if min_time_diff < 300 else None

    def _find_closest_smooth_point(self, smooth_route: List[Tuple[float, float]], 
                                 point: GeoPoint) -> int:
        """Find the index of the smooth route point closest to the given point."""
        return min(
            range(len(smooth_route)),
            key=lambda x: GeoPoint(*smooth_route[x]).distance_to(point)
        )

    def _add_uniform_markers(self, marker_indices: Set[int], distances: List[float], 
                           total_length: float) -> None:
        """Add uniformly spaced markers along the route."""
        target_markers_uniform = min(20, max(5, int(total_length / 250))) if total_length > 0 else 2
        if target_markers_uniform > 2 and total_length > 0:
            step_length = total_length / (target_markers_uniform - 1)
            for i in range(1, target_markers_uniform - 1):
                target_dist = i * step_length
                closest_idx = min(range(len(distances)), key=lambda x: abs(distances[x] - target_dist))
                marker_indices.add(closest_idx)

    def _create_checkpoints_from_markers(self, smooth_route: List[Tuple[float, float]], 
                                    route_elevations: List[float], 
                                    marker_indices: List[int]) -> List[Dict[str, any]]:
        """Create checkpoint dictionaries from the collected marker indices."""
        checkpoints = []
        distances = self._calculate_smooth_route_distances(smooth_route)
        
        for i, idx in enumerate(marker_indices):
            point = smooth_route[idx]
            elevation = route_elevations[idx]
            photo_info = self.index_to_photo_info.get(idx)
            distance_from_start = distances[idx] if idx < len(distances) else 0

            checkpoint = self._create_single_checkpoint(
                i, idx, len(marker_indices),
                point, elevation, photo_info, distance_from_start
            )
            checkpoints.append(checkpoint)
        
        return checkpoints

    def _create_single_checkpoint(self, position: int, point_index: int, total_positions: int,
                                point: Tuple[float, float], elevation: float,
                                photo_info: Optional[Dict[str, any]],
                                distance_from_start: float) -> Dict[str, any]:
        """Create a single checkpoint dictionary."""
        # Determine if this is a photo checkpoint
        is_photo_checkpoint = photo_info is not None
        
        # Get base name and description
        if position == 0:
            point_name = "Старт"
            description = "Старт маршрута."
        elif position == total_positions - 1:
            point_name = "Финиш"
            description = "Конец маршрута."
        elif is_photo_checkpoint:  # Photo checkpoint
            point_name = "Фототочка"
            description = ""
        else:  # Regular checkpoint
            point_name = f"Точка {position+1}"
            description = ""

        photo_html = get_photo_html(point[0], point[1], 
                                local_photo_path=photo_info['path'] if is_photo_checkpoint else None)

        return {
            'point_index': point_index,
            'position': position + 1,
            'total_positions': total_positions,
            'lat': point[0],
            'lon': point[1],
            'elevation': elevation,
            'distance_from_start': distance_from_start,
            'name': point_name,
            'description': description,
            'photo_html': photo_html,
            'is_photo': is_photo_checkpoint,
            'photo_path': photo_info['path'] if is_photo_checkpoint else None
        }

    def _get_checkpoint_name_and_description(self, position: int, total_positions: int) -> Tuple[str, str]:
        """Get the name and initial description for a checkpoint based on its position."""
        if position == 0:
            return "Старт", "Старт маршрута."
        elif position == total_positions - 1:
            return "Финиш", "Конец маршрута."
        else:
            return f"Точка {position+1}", ""

    def _get_checkpoint_photo_html(self, point: Tuple[float, float], 
                                 photo_info: Optional[Dict[str, any]]) -> str:
        """Generate photo HTML for a checkpoint."""
        if photo_info:
            return get_photo_html(point[0], point[1], local_photo_path=photo_info['path'])
        return get_photo_html(point[0], point[1])