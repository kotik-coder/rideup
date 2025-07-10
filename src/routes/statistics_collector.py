from typing import List, Dict
from dataclasses import dataclass
from ui.map_helpers import print_step
from routes.route import Route
from routes.track import Track

@dataclass
class ProfilePoint:
    """Represents a point in elevation or velocity profile"""
    distance: float  # in meters
    elevation: float = None  # for elevation profile
    velocity: float = None   # for velocity profile (in m/s)

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
    """Collects and generates various statistical profiles and analyses for routes and tracks."""
    def __init__(self):
        print_step("StatisticsCollector", "StatisticsCollector initialized.")
        
    def generate_route_profiles(self, route: Route, associated_tracks: List[Track]) -> Dict[str, List[ProfilePoint]]:
        """
        Generates both elevation and velocity profiles for a route.
        Returns dictionary containing both profiles as lists of ProfilePoint objects.
        """
        profiles = {
            'elevation_profile': self.create_elevation_profile(route),
            'velocity_profile': self.create_velocity_profile_from_track(associated_tracks) if associated_tracks else []
        }
        return profiles

    def create_elevation_profile(self, route: Route) -> List[ProfilePoint]:
        """
        Creates an elevation profile from a route.
        The profile consists of ProfilePoint objects with distance (cumulative) and elevation.
        Also calculates and sets the route's total_distance.

        Args:
            route (Route): The route to process

        Returns:
            List[ProfilePoint]: A list of profile points with distance and elevation
        """
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
        """
        Creates a velocity profile from the route's primary track.
        Uses the first associated track if available.

        Args:
            associated_tracks (List[Track]): The list of tracks associated with the route.

        Returns:
            List[ProfilePoint]: A list of profile points with distance and velocity
        """
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

def get_landscape_description(
    current_elevation: float,
    segment_net_elevation_change: float,
    segment_elevation_gain: float,
    segment_elevation_loss: float,
    segment_distance: float
) -> str:
    """
    Описание ландшафта на основе текущей высоты и характеристик сегмента.
    
    Args:
        current_elevation: Текущая высота в точке чекпоинта.
        segment_net_elevation_change: Чистое изменение высоты по сегменту (конец - начало).
        segment_elevation_gain: Общий набор высоты по сегменту.
        segment_elevation_loss: Общий спуск высоты по сегменту.
        segment_distance: Длина сегмента в метрах.
    """
    desc = ""
    
    # 1. Общее описание высоты точки
    if current_elevation < 100:
        desc += "Низменность. "
    elif 100 <= current_elevation < 200:
        desc += "Равнинная местность. "
    else:
        desc += "Возвышенность. "

    # 2. Описание перепада высот по сегменту
    if segment_distance > 0: # Avoid division by zero
        # Consider average gradient for more nuanced description
        avg_gradient_gain = (segment_elevation_gain / segment_distance) * 100 if segment_distance > 0 else 0
        avg_gradient_loss = (segment_elevation_loss / segment_distance) * 100 if segment_distance > 0 else 0

        # Heuristics for "ravines" or "undulating"
        # If both gain and loss are significant relative to segment length and net change
        if segment_elevation_gain > 15 and segment_elevation_loss > 15 and \
           abs(segment_net_elevation_change) < (segment_elevation_gain + segment_elevation_loss) * 0.5:
            desc += "Пересеченная местность с оврагами или чередованием подъемов и спусков. "
        elif segment_elevation_gain > 20:
            desc += "Значительный подъем. "
        elif segment_elevation_loss > 20:
            desc += "Значительный спуск. "
        elif segment_net_elevation_change > 5:
            desc += "Легкий подъем. "
        elif segment_net_elevation_change < -5:
            desc += "Легкий спуск. "
        else:
            desc += "Пологий участок. "
    else:
        desc += "Короткий, ровный участок. "
            
    return desc