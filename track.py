# track.py
import numpy as np
from scipy.signal import medfilt, savgol_filter
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
from route import GeoPoint, Route
from map_helpers import print_step

# Constants for MTB physics (adjust based on your riding style)
MAX_PLAUSIBLE_SPEED = 47.2  # m/s (~170 km/h, beyond downhill world record)
MAX_PLAUSIBLE_ACCEL = 10.0  # m/s² (aggressive MTB acceleration)
MIN_TIME_DELTA = 0.1  # seconds (ignore GPS points too close in time)

@dataclass
class TrackPoint:
    point: GeoPoint
    timestamp: datetime
    elapsed_seconds: float = 0.0

@dataclass
class TrackAnalysis:
    speed: float  # m/s
    acceleration: float  # m/s²
    distance_from_start: float  # meters

@dataclass
class Track:
    def __init__(self, points: List[TrackPoint], route: Optional[Route] = None):
        self.points = points
        self.route = route  # Direct reference to the parent route
        self.analysis: List[TrackAnalysis] = []
        self._analyze()
        
    def _filter_implausible_points(self, times: np.ndarray, distances: np.ndarray) -> tuple:
        """Remove points causing unrealistic speeds/accelerations."""
        valid_indices = []
        prev_valid_idx = 0
        
        for i in range(1, len(times)):
            dt = times[i] - times[prev_valid_idx]
            if dt < MIN_TIME_DELTA:
                continue  # Skip too-close points
            
            dd = distances[i] - distances[prev_valid_idx]
            raw_speed = dd / dt
            
            if raw_speed <= MAX_PLAUSIBLE_SPEED:
                valid_indices.append(i)
                prev_valid_idx = i
        
        return times[valid_indices], distances[valid_indices]

    def _calculate_robust_speeds(self, times: np.ndarray, distances: np.ndarray) -> tuple:
        """Central difference + median + adaptive Savitzky-Golay."""
        # Step 1: Central difference with edge handling
        speeds = np.zeros_like(distances)
        if len(times) >= 2:
            speeds[1:-1] = (distances[2:] - distances[:-2]) / (times[2:] - times[:-2])  # Central
            speeds[0] = (distances[1] - distances[0]) / (times[1] - times[0])  # Forward
            speeds[-1] = (distances[-1] - distances[-2]) / (times[-1] - times[-2])  # Backward
        
        # Step 2: Median filter to kill spikes
        speeds = medfilt(speeds, kernel_size=5 if len(speeds) >= 5 else 3)
        
        # Step 3: Cap speeds to physical limits
        speeds = np.clip(speeds, 0.0, MAX_PLAUSIBLE_SPEED)
        
        # Step 4: Adaptive Savitzky-Golay (skip if too noisy)
        if len(speeds) > 10:
            try:
                speeds = savgol_filter(speeds, window_length=min(11, len(speeds)), polyorder=2)
            except:
                pass  # Fallback to median-filtered speeds
        
        # Step 5: Calculate acceleration with sanity checks
        accelerations = np.zeros_like(speeds)
        if len(speeds) >= 2:
            accelerations[1:-1] = (speeds[2:] - speeds[:-2]) / (times[2:] - times[:-2])
            accelerations = np.clip(accelerations, -MAX_PLAUSIBLE_ACCEL, MAX_PLAUSIBLE_ACCEL)
        
        return speeds, accelerations

    def _analyze(self):
        if len(self.points) < 2:
            self.analysis = [TrackAnalysis(0.0, 0.0, 0.0) for _ in self.points]
            return

        # Prepare data
        times = np.array([p.elapsed_seconds for p in self.points])
        distances = np.cumsum([0.0] + [self.points[i].point.distance_to(self.points[i+1].point) 
                              for i in range(len(self.points)-1)])
        
        # Step 0: Pre-filter implausible points (e.g., GPS glitches)
        filtered_times, filtered_distances = self._filter_implausible_points(times, distances)
        
        if len(filtered_times) < 2:
            print("Warning: Not enough valid points after filtering")
            self.analysis = [TrackAnalysis(0.0, 0.0, distances[i]) for i in range(len(times))]
            return
        
        # Step 1: Compute speeds/accelerations
        speeds, accelerations = self._calculate_robust_speeds(filtered_times, filtered_distances)
        
        # Step 2: Map back to original points (interpolate if needed)
        final_speeds = np.interp(times, filtered_times, speeds)
        final_accels = np.interp(times, filtered_times, accelerations)
        
        self.analysis = [
            TrackAnalysis(
                speed=final_speeds[i],
                acceleration=final_accels[i],
                distance_from_start=distances[i]
            )
            for i in range(len(self.points))
        ]