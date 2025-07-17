from dataclasses import dataclass
from functools import reduce
import math
from typing import Final, List, Dict, Set, Optional, Tuple
from datetime import datetime, timezone

from src.routes.route import GeoPoint
from src.iio.spot_photo import SpotPhoto
from src.routes.track import Track
from src.iio.media_helpers import get_photo_html # Assuming this function exists and is correctly imported

def closest_route_point(a : GeoPoint, route : List[GeoPoint]) -> int:
    return min(
        range(len(route)),
        key=lambda i: route[i].distance_to(a.point)
    )

class Checkpoint:
    
    name : str = "Точка"
    checkpoint_index : int
    route_point_index : int
    point : GeoPoint
    distance_from_origin : float    
    description : str
    photo_html : str
    
    """
    Represents a single checkpoint on a route.
    """
    def __init__(self,
                 point_index: int,
                 route_point_index : int,
                 point : GeoPoint):
        self.checkpoint_index = point_index
        self.route_point_index = route_point_index
        self.point = point
        self.description = ""
        self.init_photo_info()
            
    def init_photo_info(self):
        self.photo_html = get_photo_html(self.point.lat, self.point.lon)
        
    def to_dict(self):
        """Converts the Checkpoint object to a dictionary."""
        return {
            'point_index': self.checkpoint_index,
            'point': self.point.to_dict,            
        }

        
class PhotoCheckpoint(Checkpoint):
    
    photo_path : str
        
    """
    Represents a single checkpoint on a route.
    """
    #@override
    def __init__(self,
                 point_index: int,
                 route_point_index: int,
                 point : GeoPoint,
                 photo_path: str):
        self.photo_path = photo_path        
        super().__init__(point_index = point_index, 
                         route_point_index = route_point_index,
                         point = point)                
    
    #@override
    def init_photo_info(self):
        self.photo_html = get_photo_html(self.point.lat, 
                                         self.point.lon,
                                         local_photo_path=self.photo_path)
        
    def to_dict(self):
        """Converts the Checkpoint object to a dictionary."""
        return {
            'point_index': self.checkpoint_index,
            'point': self.point.to_dict,            
            'photo_path': self.point.to_dict,            
        }
    
@dataclass
class CheckpointGenerator:
    
    local_photos : List[SpotPhoto]
    
    MAX_DELAY      : Final[float] = 60     # seconds    
    TARGET_DENSITY : Final[float] = 30 / 5000 #30 checkpoints per 5,000 metres
        
    def generate_checkpoints(self,
                             smooth_points: List[GeoPoint],
                             associated_tracks: List[Track]) -> List[Checkpoint]:
        
        distances = [0.0]
        for i in range(1, len(smooth_points)):
            p1 = smooth_points[i-1]
            p2 = smooth_points[i]
            distances.append(distances[-1] + p1.distance_to(p2))

        #ensure both endpoints are always present
        marker_indices = [0, len(smooth_points) - 1]

        #priority for user-uploaded photo checkpoints
        photo_checkpoints_data = self._add_photo_markers(smooth_points, associated_tracks)                
        photo_markers = list(photo_checkpoints_data.keys())
        
        marker_indices.sort()
        marker_indices.extend(photo_markers)    
        
        #fill gaps with uniform markers
        self._add_uniform_markers(marker_indices, smooth_points, distances)    
        
        #combine both        
        return self._create_checkpoints_from_markers(
            smooth_points,
            marker_indices,
            photo_checkpoints_data,
            distances
        )

    def _add_photo_markers(self,
                           route: List[GeoPoint],
                           associated_tracks: List[Track]) -> Dict[int, SpotPhoto]:
        
        photopoint_dict: Dict[int, SpotPhoto] = {}
        
        for track in associated_tracks:
            if not track.points:
                continue

            # Sort photos by timestamp to process chronologically
            for photo in sorted(self.local_photos, 
                                key=lambda x: x.timestamp if x.timestamp else datetime.min.replace(tzinfo=timezone.utc)
                                ):
                
                # Ensure photo has a timestamp and has not been processed already in this run
                if photo.timestamp:
                    
                    self._process_photo_for_marker(
                        photo, track, route,
                        photopoint_dict # Pass the accumulating list
                    )

        return photopoint_dict

    def _process_photo_for_marker(self,
                                  photo: SpotPhoto,
                                  track: Track,
                                  route: List[GeoPoint],
                                  current_photo_checkpoints: Dict[int, SpotPhoto]) -> Optional[Tuple[GeoPoint, int, SpotPhoto]]:

        track_start_time = track.points[0].timestamp #start time
        
        photo_time = photo.timestamp
        if not photo_time:
            return None

        photo_elapsed = (photo_time - track_start_time).total_seconds()

        closest_track_point = track.find_closest_track_point(photo_elapsed, self.MAX_DELAY)

        if closest_track_point is None \
           or abs(photo_elapsed - closest_track_point.elapsed_seconds) >= self.MAX_DELAY:
            return None

        closest_index = closest_route_point(closest_track_point, route)
        closest_point = route[closest_index]
        
        result = None
        
        if closest_index not in current_photo_checkpoints:
            current_photo_checkpoints[closest_index] = photo        
            result = closest_point, photo
        
        return result
        
    def _add_uniform_markers(self,
                             marker_indices: List[int],
                             smooth_points : List[GeoPoint],
                             route_distances : List[float]):
            
        if sorted(route_distances) != route_distances:
            print("Error: distances not monotonic!")
            return

        total_length = route_distances[-1]   
                # Determine maximum allowed markers based on 20 per 5 km rule
        max_markers = min( 
                          int(total_length * self.TARGET_DENSITY), 
                          len(smooth_points))
        
        num_markers = len(marker_indices)     
        
        while num_markers < max_markers:
            
            # Calculate distances between consecutive checkpoints
            gaps = [
                (route_distances[b] - route_distances[a], a, b)
                for a, b in zip(marker_indices, marker_indices[1:])
            ]

            large_gaps    = [g for g in gaps if g[0] > self.TARGET_DENSITY * total_length]
            max_gap_count = min ( max_markers - num_markers, len(large_gaps) )                        
            
            for i in range(max_gap_count):
                
                gap = large_gaps[i]                
                marker_indices.append( (gap[1] + gap[2]) // 2 )

            marker_indices.sort()
            num_markers += max_gap_count

    def _create_checkpoints_from_markers(self,
                                         smooth_points : List[GeoPoint],
                                         marker_indices: List[int],
                                         photo_markers: Dict[int, SpotPhoto],
                                         distances : List[float]) -> List[Checkpoint]:

        checkpoints = []
        total_indices = len(marker_indices)        

        for i, idx in enumerate(marker_indices):
            point      = smooth_points[idx]
                        
            photo_path = None
            if idx in photo_markers:
                photo_info = photo_markers[idx] # Get SpotPhoto object                        
                photo_path = photo_info.fname

            if photo_path:                
                checkpoint = PhotoCheckpoint(i, idx, point, photo_path)                
                checkpoint.name = f"Фототочка ({i + 1}/{total_indices})"
            else:
                checkpoint = Checkpoint(i, idx, point)
                checkpoint.name = f"Точка ({i + 1}/{total_indices})"                                

            checkpoint.distance_from_origin = distances[idx]

            checkpoints.append(checkpoint)
            
        checkpoints[0].name  = "Старт"
        checkpoints[-1].name = "Финиш"

        return checkpoints