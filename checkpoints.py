from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime, timezone
from route import GeoPoint
from track import Track, TrackPoint
from media_helpers import get_photo_html # Assuming this function exists and is correctly imported

PHOTO_CHECKPOINT_DISTANCE_THRESHOLD = 50.0  # meters

class Checkpoint:
    """
    Represents a single checkpoint on a route.
    """
    def __init__(self,
                 point_index: int,
                 position: int,
                 total_positions: int,
                 lat: float,
                 lon: float,
                 elevation: float,
                 distance_from_start: float,
                 name: str,
                 description: str,
                 is_photo: bool = False,
                 photo_path: Optional[str] = None):
        self.point_index = point_index
        self.position = position
        self.total_positions = total_positions
        self.lat = lat
        self.lon = lon
        self.elevation = elevation
        self.distance_from_start = distance_from_start
        self.name = name
        self.description = description
        self.is_photo = is_photo
        self.photo_path = photo_path
        self.photo_html = self._generate_photo_html()

    def _generate_photo_html(self) -> str:
        """Generates the HTML for the photo associated with this checkpoint, if any."""
        if self.is_photo and self.photo_path:
            return get_photo_html(self.lat, self.lon, local_photo_path=self.photo_path)
        return get_photo_html(self.lat, self.lon) # Return default HTML if not a photo checkpoint

    def to_dict(self) -> Dict[str, any]:
        """Converts the Checkpoint object to a dictionary."""
        return {
            'point_index': self.point_index,
            'position': self.position,
            'total_positions': self.total_positions,
            'lat': self.lat,
            'lon': self.lon,
            'elevation': self.elevation,
            'distance_from_start': self.distance_from_start,
            'name': self.name,
            'description': self.description,
            'photo_html': self.photo_html,
            'is_photo': self.is_photo,
            'photo_path': self.photo_path
        }

class CheckpointGenerator:
    """
    Generates checkpoints along a smooth route, including start, end,
    uniformly spaced points, and points associated with local photos.
    """
    def __init__(self, local_photos: List[Dict[str, any]]):
        self.local_photos = local_photos
        # Maps smooth route index to photo info for photo checkpoints
        self.index_to_photo_info: Dict[int, Dict[str, any]] = {}

    def generate_checkpoints(self,
                             smooth_route: List[Tuple[float, float]],
                             route_elevations: List[float],
                             associated_tracks: List[Track]) -> List[Checkpoint]:
        """
        Main method to generate all checkpoints for a route.
        Args:
            smooth_route: A list of (latitude, longitude) tuples representing the smoothed route.
            route_elevations: A list of elevation values corresponding to each point in smooth_route.
            associated_tracks: A list of Track objects associated with the route, used for photo matching.
        Returns:
            A list of Checkpoint objects.
        """
        if len(smooth_route) < 2:
            return []

        distances = self._calculate_smooth_route_distances(smooth_route)
        total_length = distances[-1] if distances else 0

        marker_indices = self._collect_marker_indices(
            smooth_route,
            total_length,
            distances,
            associated_tracks
        )

        return self._create_checkpoints_from_markers(
            smooth_route,
            route_elevations,
            distances,
            marker_indices
        )

    def _calculate_smooth_route_distances(self, smooth_route: List[Tuple[float, float]]) -> List[float]:
        """
        Calculate cumulative distances along the smooth route.
        Args:
            smooth_route: A list of (latitude, longitude) tuples.
        Returns:
            A list where each element is the cumulative distance from the start of the route
            to the corresponding point in smooth_route.
        """
        distances = [0.0]
        for i in range(1, len(smooth_route)):
            p1 = GeoPoint(*smooth_route[i-1])
            p2 = GeoPoint(*smooth_route[i])
            distances.append(distances[-1] + p1.distance_to(p2))
        return distances

    def _collect_marker_indices(self,
                                smooth_route: List[Tuple[float, float]],
                                total_length: float,
                                distances: List[float],
                                associated_tracks: List[Track]) -> List[int]:
        """
        Collects all unique indices from the smooth route that should be marked as checkpoints.
        Includes start, end, photo points, and uniformly spaced points.
        Args:
            smooth_route: The smoothed route points.
            total_length: The total length of the route.
            distances: Cumulative distances for each point in smooth_route.
            associated_tracks: Tracks used to find photo locations.
        Returns:
            A sorted list of unique indices for checkpoints.
        """
        marker_indices = set()
        marker_indices.add(0) # Start point
        marker_indices.add(len(smooth_route) - 1) # End point

        self._add_photo_markers(marker_indices, smooth_route, associated_tracks)
        self._add_uniform_markers(marker_indices, distances, total_length)

        return sorted(list(marker_indices))

    def _is_photo_already_processed(self, photo_path: str) -> bool:
        """
        Checks if a photo with the given path has already been processed and added
        as a photo checkpoint.
        Args:
            photo_path: The file path of the photo.
        Returns:
            True if the photo has already been processed, False otherwise.
        """
        return photo_path in [p.get('path') for p in self.index_to_photo_info.values()]

    def _add_photo_markers(self,
                           marker_indices: Set[int],
                           smooth_route: List[Tuple[float, float]],
                           associated_tracks: List[Track]) -> None:
        """
        Adds indices for points near local photos to the marker_indices set.
        Args:
            marker_indices: The set to add marker indices to.
            smooth_route: The smoothed route points.
            associated_tracks: Tracks containing timestamped points for photo matching.
        """
        for track in associated_tracks:
            if not track.points:
                continue

            track_start_time = track.points[0].timestamp
            track_end_time = track.points[-1].timestamp
            track_duration = (track_end_time - track_start_time).total_seconds()

            for photo in sorted(self.local_photos, key=lambda x: x.get('timestamp', datetime.min.replace(tzinfo=timezone.utc))):
                # Ensure photo has a timestamp and has not been processed already
                if photo.get('timestamp') and not self._is_photo_already_processed(photo['path']):
                    self._process_photo_for_marker(
                        photo, track, track_start_time,
                        track_duration, smooth_route,
                        marker_indices
                    )

    def _process_photo_for_marker(self,
                                  photo: Dict[str, any],
                                  track: Track,
                                  track_start_time: datetime,
                                  track_duration: float,
                                  smooth_route: List[Tuple[float, float]],
                                  marker_indices: Set[int]) -> None:
        """
        Processes a single photo to determine if it should add a marker.
        Args:
            photo: Dictionary containing photo information (must have 'timestamp' and 'path').
            track: The current track being processed.
            track_start_time: Start timestamp of the track.
            track_duration: Duration of the track in seconds.
            smooth_route: The smoothed route points.
            marker_indices: The set to add marker indices to.
        """
        photo_time = photo.get('timestamp')
        if not photo_time:
            return

        photo_elapsed = (photo_time - track_start_time).total_seconds()

        # Check if photo time is within track duration
        if not (0 <= photo_elapsed <= track_duration):
            return

        # Find the closest track point by elapsed time
        closest_track_point = self._find_closest_track_point(track.points, photo_elapsed)

        # If no suitable track point found or time difference is too large
        if closest_track_point is None or abs(photo_elapsed - closest_track_point.elapsed_seconds) >= 300:
            return

        # Find the closest smooth route point to the track point's geographic coordinates
        closest_smooth_idx = self._find_closest_smooth_point(smooth_route, closest_track_point.point)

        # Check for proximity to existing photo checkpoints to avoid duplicates
        is_too_close_to_existing_photo = False
        for existing_idx in self.index_to_photo_info.keys():
            existing_point = GeoPoint(*smooth_route[existing_idx])
            new_point = GeoPoint(*smooth_route[closest_smooth_idx])
            if new_point.distance_to(existing_point) < PHOTO_CHECKPOINT_DISTANCE_THRESHOLD:
                is_too_close_to_existing_photo = True
                break

        if not is_too_close_to_existing_photo:
            marker_indices.add(closest_smooth_idx)
            # Store the full photo info in the dictionary, keyed by smooth route index
            self.index_to_photo_info[closest_smooth_idx] = {
                'path': photo['path'],
                'timestamp': photo_time,
                'coords': photo.get('coords') # Store original photo coords if available
            }

    def _find_closest_track_point(self,
                                  track_points: List[TrackPoint],
                                  photo_elapsed: float) -> Optional[TrackPoint]:
        """
        Finds the track point closest in elapsed time to the photo's timestamp.
        Args:
            track_points: List of TrackPoint objects.
            photo_elapsed: Elapsed time of the photo from the track's start.
        Returns:
            The TrackPoint closest to the photo's elapsed time, or None if no suitable point is found.
        """
        closest_track_point = None
        min_time_diff = float('inf')

        # Use binary search or iterate for efficiency if track_points is large and sorted by time
        for tp in track_points:
            time_diff = abs(photo_elapsed - tp.elapsed_seconds)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_track_point = tp

        # Only return if the closest point is within a reasonable time difference (e.g., 5 minutes)
        return closest_track_point if min_time_diff < 300 else None

    def _find_closest_smooth_point(self,
                                 smooth_route: List[Tuple[float, float]],
                                 point: GeoPoint) -> int:
        """
        Finds the index of the smooth route point closest to the given geographic point.
        Args:
            smooth_route: List of (latitude, longitude) tuples for the smooth route.
            point: The GeoPoint to find the closest smooth route point for.
        Returns:
            The index of the closest point in smooth_route.
        """
        return min(
            range(len(smooth_route)),
            key=lambda i: GeoPoint(*smooth_route[i]).distance_to(point)
        )

    def _add_uniform_markers(self,
                           marker_indices: Set[int],
                           distances: List[float],
                           total_length: float) -> None:
        """
        Adds uniformly spaced markers along the route to the marker_indices set.
        Args:
            marker_indices: The set to add marker indices to.
            distances: Cumulative distances for each point in smooth_route.
            total_length: The total length of the route.
        """
        # Aim for a reasonable number of uniform markers, between 5 and 20,
        # approximately every 250 meters if the route is long enough.
        target_markers_uniform = min(20, max(5, int(total_length / 250))) if total_length > 0 else 2

        if target_markers_uniform > 2 and total_length > 0:
            step_length = total_length / (target_markers_uniform - 1)
            for i in range(1, target_markers_uniform - 1): # Exclude start and end, already added
                target_dist = i * step_length
                # Find the index of the point closest to the target distance
                closest_idx = min(range(len(distances)), key=lambda x: abs(distances[x] - target_dist))
                marker_indices.add(closest_idx)

    def _create_checkpoints_from_markers(self,
                                         smooth_route: List[Tuple[float, float]],
                                         route_elevations: List[float],
                                         distances: List[float],
                                         marker_indices: List[int]) -> List[Checkpoint]:
        """
        Creates Checkpoint objects from the collected marker indices.
        Args:
            smooth_route: The smoothed route points.
            route_elevations: Elevations corresponding to smooth_route points.
            distances: Cumulative distances for smooth_route points.
            marker_indices: Sorted list of indices to create checkpoints for.
        Returns:
            A list of Checkpoint objects.
        """
        checkpoints = []
        total_positions = len(marker_indices)

        for i, idx in enumerate(marker_indices):
            point = smooth_route[idx]
            elevation = route_elevations[idx]
            photo_info = self.index_to_photo_info.get(idx)
            distance_from_start = distances[idx] if idx < len(distances) else 0

            is_photo_checkpoint = photo_info is not None
            photo_path = photo_info['path'] if is_photo_checkpoint else None

            # Determine checkpoint name and description
            if i == 0:
                name = "Старт"
                description = "Старт маршрута."
            elif i == total_positions - 1:
                name = "Финиш"
                description = "Конец маршрута."
            elif is_photo_checkpoint:
                name = "Фототочка"
                description = ""
            else:
                name = f"Точка {i+1}"
                description = ""

            checkpoint = Checkpoint(
                point_index=idx,
                position=i + 1, # 1-based position
                total_positions=total_positions,
                lat=point[0],
                lon=point[1],
                elevation=elevation,
                distance_from_start=distance_from_start,
                name=name,
                description=description,
                is_photo=is_photo_checkpoint,
                photo_path=photo_path
            )
            checkpoints.append(checkpoint)

        return checkpoints
