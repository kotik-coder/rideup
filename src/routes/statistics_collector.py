from typing import List, Dict
from dataclasses import dataclass

from src.ui.map_helpers import print_step
from src.routes.route import Route
from src.routes.track import Track

@dataclass
class ProfilePoint:
    """Represents a point in elevation or velocity profile"""
    distance:  float        # in meters
    elevation: float = 0.0  # for elevation profile
    velocity:  float = 0.0  # for velocity profile (in m/s)

@dataclass
class Segment:
    """Represents a route segment between two checkpoints"""
    distance: float  # in meters
    elevation_gain: float
    elevation_loss: float
    net_elevation: float
    min_elevation: float
    max_elevation: float
    start_checkpoint_index: int
    end_checkpoint_index: int

class StatisticsCollector:

    def __init__(self):
        print_step("StatisticsCollector", "StatisticsCollector initialized.")
        
    def generate_route_profiles(self, route: Route, associated_tracks: List[Track]) -> Dict[str, List[ProfilePoint]]:
        
        profiles = {
            'elevation_profile': self.create_elevation_profile(route),
            'velocity_profile': self.create_velocity_profile_from_track(associated_tracks) if associated_tracks else []
        }
        return profiles

    def create_elevation_profile(self, route: Route) -> List[ProfilePoint]:
        
        profile = []
        route.total_distance = 0.0  # Initialize total distance
        
        if not route.points or not route.elevations or len(route.points) != len(route.elevations):
            print_step("StatisticsCollector", "Elevation profile creation: Empty or mismatched points/elevations. Returning empty profile.", level="WARNING")
            return []

        # Add the starting point
        profile.append(ProfilePoint(distance=0.0, elevation=route.elevations[0]))
        
        # Calculate cumulative distance and add subsequent points
        for i in range(1, len(route.points)):
            segment_distance = route.points[i-1].distance_to(route.points[i])
            route.total_distance += segment_distance
            profile.append(ProfilePoint(distance=route.total_distance, elevation=route.elevations[i]))
        
        print_step("StatisticsCollector", f"Elevation profile created with {len(profile)} points. Total distance: {route.total_distance:.2f}m")
        return profile

    def create_velocity_profile_from_track(self, associated_tracks: List[Track]) -> List[ProfilePoint]:
        
        if not associated_tracks:
            print_step("StatisticsCollector", "No tracks available for velocity profile.", level="WARNING")
            return []

        track = associated_tracks[0]  # Use the first associated track
        profile = []
        
        if not track.analysis:
            print_step("StatisticsCollector", "Velocity profile creation: Track has no analysis data. Returning empty profile.")
            return []

        for analysis_point in track.analysis:
            profile.append(ProfilePoint(
                distance=analysis_point.distance_from_start,
                velocity=analysis_point.speed
            ))
        
        print_step("StatisticsCollector", f"Velocity profile created successfully ({len(profile)} points).")
        return profile