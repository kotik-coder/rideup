from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime, timezone

from src.routes.route import GeoPoint
from src.iio.spot_photo import SpotPhoto
from src.routes.track import Track, TrackPoint
from src.iio.media_helpers import get_photo_html # Assuming this function exists and is correctly imported

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
        result = ""        
        if self.is_photo and self.photo_path:            
            result = get_photo_html(self.lat, self.lon, local_photo_path=self.photo_path)
        else:
            result = get_photo_html(self.lat, self.lon) # Return default HTML if not a photo checkpoint
        return result

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
    def __init__(self, local_photos: List[SpotPhoto]): # Changed type hint here
        self.local_photos = local_photos

    def generate_checkpoints(self,
                             smooth_route: List[GeoPoint],
                             associated_tracks: List[Track]) -> List[Checkpoint]:
        """
        Main method to generate all checkpoints for a route.
        """
        if len(smooth_route) < 2:
            return []

        distances = self._calculate_smooth_route_distances(smooth_route)
        total_length = distances[-1] if distances else 0

        # Now _collect_marker_indices returns both marker indices and photo checkpoint data
        marker_indices, photo_checkpoints_data = self._collect_marker_indices(
            smooth_route,
            total_length,
            distances,
            associated_tracks
        )

        return self._create_checkpoints_from_markers(
            smooth_route,
            distances,
            marker_indices,
            photo_checkpoints_data # Pass photo checkpoint data
        )

    def _calculate_smooth_route_distances(self, smooth_route: List[GeoPoint]) -> List[float]:

        distances = [0.0]
        for i in range(1, len(smooth_route)):
            p1 = smooth_route[i-1]
            p2 = smooth_route[i]
            distances.append(distances[-1] + p1.distance_to(p2))
        return distances

    def _collect_marker_indices(self,
                                smooth_route: List[GeoPoint],
                                total_length: float,
                                distances: List[float],
                                associated_tracks: List[Track]) -> Tuple[List[int], List[Tuple[int, SpotPhoto]]]:
        """
        Collects all unique indices from the smooth route that should be marked as checkpoints,
        and also returns the associated photo data.
        Returns:
            A tuple: (sorted list of unique indices for checkpoints,
                      list of (index, SpotPhoto) for photo checkpoints).
        """
        marker_indices = set()
        marker_indices.add(0) # Start point
        marker_indices.add(len(smooth_route) - 1) # End point

        # photo_checkpoints_data will be populated and returned by _add_photo_markers
        photo_checkpoints_data = self._add_photo_markers(marker_indices, smooth_route, associated_tracks)
        
        self._add_uniform_markers(marker_indices, distances, total_length)

        return sorted(list(marker_indices)), photo_checkpoints_data

    def _add_photo_markers(self,
                           marker_indices: Set[int],
                           smooth_route: List[GeoPoint],
                           associated_tracks: List[Track]) -> List[Tuple[int, SpotPhoto]]:
        """
        Adds indices for points near local photos to the marker_indices set, and
        returns a list of (index, SpotPhoto) for the photo checkpoints identified.
        Args:
            marker_indices: The set to add marker indices to.
            smooth_route: The smoothed route points.
            associated_tracks: Tracks containing timestamped points for photo matching.
        Returns:
            A list of tuples (index, SpotPhoto) for all identified photo checkpoints.
        """
        identified_photo_checkpoints: List[Tuple[int, SpotPhoto]] = []
        
        # Keep track of photo paths already processed in this run to avoid duplicates
        processed_photo_paths: Set[str] = set() 

        for track in associated_tracks:
            if not track.points:
                continue

            track_start_time = track.points[0].timestamp
            track_end_time = track.points[-1].timestamp
            track_duration = (track_end_time - track_start_time).total_seconds()

            # Sort photos by timestamp to process chronologically
            for photo in sorted(self.local_photos, key=lambda x: x.timestamp if x.timestamp else datetime.min.replace(tzinfo=timezone.utc)):
                # Ensure photo has a timestamp and has not been processed already in this run
                if photo.timestamp and photo.path not in processed_photo_paths:
                    result = self._process_photo_for_marker(
                        photo, track, track_start_time,
                        track_duration, smooth_route,
                        identified_photo_checkpoints # Pass the accumulating list
                    )
                    if result:
                        smooth_idx, spot_photo_obj = result
                        marker_indices.add(smooth_idx)
                        identified_photo_checkpoints.append((smooth_idx, spot_photo_obj))
                        processed_photo_paths.add(spot_photo_obj.path) # Mark photo as processed

        return identified_photo_checkpoints

    def _process_photo_for_marker(self,
                                  photo: SpotPhoto,
                                  track: Track,
                                  track_start_time: datetime,
                                  track_duration: float,
                                  smooth_route: List[GeoPoint],
                                  current_photo_checkpoints: List[Tuple[int, SpotPhoto]]) -> Optional[Tuple[int, SpotPhoto]]:
        """
        Processes a single photo to determine if it should add a marker.
        Args:
            photo: SpotPhoto object containing photo information.
            track: The current track being processed.
            track_start_time: Start timestamp of the track.
            track_duration: Duration of the track in seconds.
            smooth_route: The smoothed route points.
            current_photo_checkpoints: List of (index, SpotPhoto) tuples already identified as photo checkpoints in this run.
        Returns:
            A tuple (closest_smooth_idx, photo) if a photo checkpoint is identified, otherwise None.
        """
        photo_time = photo.timestamp
        if not photo_time:
            return None

        photo_elapsed = (photo_time - track_start_time).total_seconds()

        if not (0 <= photo_elapsed <= track_duration):
            return None

        closest_track_point = self._find_closest_track_point(track.points, photo_elapsed)

        if closest_track_point is None or abs(photo_elapsed - closest_track_point.elapsed_seconds) >= 300:
            return None

        closest_smooth_idx = self._find_closest_smooth_point(smooth_route, closest_track_point.point)

        # Check for proximity to existing photo checkpoints identified so far
        is_too_close_to_existing_photo = False
        for existing_idx, _ in current_photo_checkpoints: # Iterate through the passed list
            existing_point = smooth_route[existing_idx]
            new_point = smooth_route[closest_smooth_idx]
            if new_point.distance_to(existing_point) < PHOTO_CHECKPOINT_DISTANCE_THRESHOLD:
                is_too_close_to_existing_photo = True
                break

        if not is_too_close_to_existing_photo:
            # Return the index and the SpotPhoto object
            return closest_smooth_idx, photo
        
        return None

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
                                 smooth_route: List[GeoPoint],
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
            key=lambda i: smooth_route[i].distance_to(point)
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
                                         smooth_route: List[GeoPoint],
                                         distances: List[float],
                                         marker_indices: List[int],
                                         photo_checkpoints_data: List[Tuple[int, SpotPhoto]]) -> List[Checkpoint]:

        checkpoints = []
        total_positions = len(marker_indices)

        # Create a temporary mapping for efficient lookup of photo info by index
        # This temporary dictionary is created inside the method and is not a class attribute.
        idx_to_photo: Dict[int, SpotPhoto] = {idx: photo for idx, photo in photo_checkpoints_data}

        for i, idx in enumerate(marker_indices):
            point     = smooth_route[idx]
            elevation = smooth_route[idx].elevation
            photo_info = idx_to_photo.get(idx) # Get SpotPhoto object
            distance_from_start = distances[idx] if idx < len(distances) else 0

            is_photo_checkpoint = photo_info is not None
            photo_path = photo_info.fname if is_photo_checkpoint else None # Access attribute directly

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
                name = f"Точка  {i+1}"
                description = ""

            checkpoint = Checkpoint(
                point_index=idx,
                position=i + 1, # 1-based position
                total_positions=total_positions,
                lat=point.lat,
                lon=point.lon,
                elevation=elevation,
                distance_from_start=distance_from_start,
                name=name,
                description=description,
                is_photo=is_photo_checkpoint,
                photo_path=photo_path
            )
            checkpoints.append(checkpoint)

        return checkpoints