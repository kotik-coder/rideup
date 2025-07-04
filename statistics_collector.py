from typing import List, Dict, Tuple
from dataclasses import dataclass
from map_helpers import print_step
from route import GeoPoint, Route
from track import Track

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
        
    def generate_route_profiles(self, route: Route, associated_tracks: List[Track]) -> Dict[str, any]:
        """
        Generates both elevation and velocity profiles for a route.
        Returns dictionary containing both profiles.
        """
        profiles = {
            'elevation_profile': self.create_elevation_profile(
                [(p.lat, p.lon) for p in route.points],
                route.elevations
            ) if route.points and route.elevations else [],
            'velocity_profile': self.create_velocity_profile_from_track(associated_tracks[0]) if associated_tracks else []
        }
        return profiles


    def create_elevation_profile(self, points: List[Tuple[float, float]], elevations: List[float]) -> List[Dict[str, float]]:
        """
        Creates an elevation profile from a list of points and their corresponding elevations.
        The profile consists of dictionaries with 'distance' (cumulative) and 'elevation'.

        Args:
            points (List[Tuple[float, float]]): List of (latitude, longitude) tuples.
            elevations (List[float]): List of elevation values corresponding to the points.

        Returns:
            List[Dict[str, float]]: A list of dictionaries, each with 'distance' and 'elevation'.
        """
        profile = []
        total_distance = 0.0
        
        if not points or not elevations or len(points) != len(elevations):
            print_step("StatisticsCollector", "Elevation profile creation: Empty or mismatched points/elevations. Returning empty profile.", level="WARNING")
            return []

        # Add the starting point
        profile.append({'distance': 0.0, 'elevation': elevations[0]})
        
        # Calculate cumulative distance and add subsequent points
        for i in range(1, len(points)):
            p1 = GeoPoint(points[i-1][0], points[i-1][1])
            p2 = GeoPoint(points[i][0], points[i][1])
            total_distance += p1.distance_to(p2)
            profile.append({'distance': total_distance, 'elevation': elevations[i]})
        
        print_step("StatisticsCollector", f"Elevation profile created with {len(profile)} points.")
        return profile

    def create_velocity_profile_from_track(self, track: Track) -> List[Dict[str, float]]:
        """
        Leverages the Track object's internal analysis to create a velocity profile.
        This method acts as an interface to the Track's functionality.

        Args:
            track (Track): The Track object from which to extract the velocity profile.

        Returns:
            List[Dict[str, float]]: A list of dictionaries, each with 'distance' and 'velocity'.
        """
        if not isinstance(track, Track):
            print_step("StatisticsCollector", "Invalid input: Expected a Track object for velocity profile.", level="ERROR")
            return []

        profile = []
        if not track.analysis:
            print_step("Процессинг", f"Создание профиля скорости: Трек не содержит данных анализа. Возвращаю пустой профиль.")
            return []

        for analysis_point in track.analysis:
            profile.append({
                'distance': analysis_point.distance_from_start,
                'velocity': analysis_point.speed
            })
        print_step("Процессинг", f"Профиль скорости для трека создан успешно.")

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