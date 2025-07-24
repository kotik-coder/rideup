# track.py
import numpy as np
from scipy.signal import medfilt, savgol_filter
from scipy.interpolate import PchipInterpolator
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

from src.routes.route import GeoPoint, Route

# Constants
MAX_PLAUSIBLE_SPEED = 47.2  # m/s
MAX_PLAUSIBLE_ACCEL = 10.0   # m/s²
MAX_VERTICAL_SPEED = 10.0    # m/s (unlikely to exceed this climbing/descending rate)
MIN_TIME_DELTA = 0.1         # seconds

@dataclass
class TrackPoint:
    point: GeoPoint
    timestamp: datetime
    elapsed_seconds: float = 0.0

@dataclass
class TrackAnalysis:
    horizontal_speed: float    # m/s
    vertical_speed: float      # m/s
    horizontal_accel: float    # m/s²
    vertical_accel: float      # m/s²
    distance_from_start: float # meters

class Track:

    points   : List[TrackPoint]
    route    : Optional[Route]
    analysis : List[TrackAnalysis]
    
    def __init__(self, points: List[TrackPoint], route: Optional[Route] = None):
        self.points = points
        self.route = route
        self.analysis: List[TrackAnalysis] = []
        self._analyze()
        
    def duration(self) -> float:
        return self.points[-1].elapsed_seconds - self.points[0].elapsed_seconds
        
    def _filter_implausible_points(self, times: np.ndarray, distances: np.ndarray, elevations: np.ndarray) -> tuple:
        valid_indices = [0]
        prev_valid_idx = 0
        
        for i in range(1, len(times)):
            dt = times[i] - times[prev_valid_idx]
            if dt < MIN_TIME_DELTA:
                continue
                
            dd = distances[i] - distances[prev_valid_idx]
            dh = elevations[i] - elevations[prev_valid_idx]
            
            horizontal_speed = dd / dt
            vertical_speed = dh / dt
            
            if (horizontal_speed <= MAX_PLAUSIBLE_SPEED and 
                abs(vertical_speed) <= MAX_VERTICAL_SPEED):
                valid_indices.append(i)
                prev_valid_idx = i
        
        return (times[valid_indices], 
                distances[valid_indices], 
                elevations[valid_indices])

    def _calculate_motion_components(self, times: np.ndarray, 
                                   distances: np.ndarray, 
                                   elevations: np.ndarray) -> tuple:
        """Calculate horizontal and vertical motion using PChip interpolation"""
        # Create PChip interpolators
        dist_interp = PchipInterpolator(times, distances)
        elev_interp = PchipInterpolator(times, elevations)
        
        # Calculate derivatives (velocities)
        h_speeds = dist_interp.derivative()(times)
        v_speeds = elev_interp.derivative()(times)
        
        # Smooth velocities
        h_speeds = self._smooth_series(h_speeds)
        v_speeds = self._smooth_series(v_speeds)
        
        # Calculate accelerations from smoothed velocities
        h_accels = np.gradient(h_speeds, times)
        v_accels = np.gradient(v_speeds, times)
        
        # Apply physical limits
        h_speeds = np.clip(h_speeds, 0.0, MAX_PLAUSIBLE_SPEED)
        v_speeds = np.clip(v_speeds, -MAX_VERTICAL_SPEED, MAX_VERTICAL_SPEED)
        h_accels = np.clip(h_accels, -MAX_PLAUSIBLE_ACCEL, MAX_PLAUSIBLE_ACCEL)
        v_accels = np.clip(v_accels, -MAX_PLAUSIBLE_ACCEL, MAX_PLAUSIBLE_ACCEL)
        
        return h_speeds, v_speeds, h_accels, v_accels

    def _smooth_series(self, data: np.ndarray) -> np.ndarray:
        """Apply median and Savitzky-Golay filtering"""
        if len(data) >= 5:
            data = medfilt(data, kernel_size=5)
        if len(data) > 10:
            try:
                data = savgol_filter(data, 
                                    window_length=min(11, len(data)), 
                                    polyorder=2)
            except:
                pass
        return data

    def _analyze(self):
        # Prepare data arrays
        times = np.array([p.elapsed_seconds for p in self.points])
        distances = np.cumsum([0.0] + [
            self.points[i].point.distance_to(self.points[i+1].point) 
            for i in range(len(self.points)-1)
        ])
        elevations = np.array([p.point.elevation for p in self.points])
        
        # Filter implausible points
        filtered_data = self._filter_implausible_points(times, distances, elevations)
        if len(filtered_data[0]) < 2:
            print("Warning: Not enough valid points after filtering")
            return
            
        # Calculate motion components
        h_speeds, v_speeds, h_accels, v_accels = self._calculate_motion_components(*filtered_data)
        
        # Create analysis points
        self.analysis = [
            TrackAnalysis(
                horizontal_speed=float(h_speeds[i]),
                vertical_speed=float(v_speeds[i]),
                horizontal_accel=float(h_accels[i]),
                vertical_accel=float(v_accels[i]),
                distance_from_start=float(distances[i])
            )
            for i in range(len(h_speeds))
        ]

    def find_closest_track_point(self, timestamp_seconds: float, tolerance: float) -> Optional[TrackPoint]:
        if timestamp_seconds < 0 or timestamp_seconds > self.duration():
            return None
            
        closest = [p for p in self.points 
                  if abs(p.elapsed_seconds - timestamp_seconds) < tolerance]
        closest.sort(key=lambda p: abs(p.elapsed_seconds - timestamp_seconds))
        return closest[0] if closest else None