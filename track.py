# track.py
import numpy as np
from scipy.interpolate import Akima1DInterpolator, interp1d # Added interp1d for linear interpolation back
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timezone
from route import GeoPoint

@dataclass
class TrackPoint:
    """Track point with timestamp and dynamic data"""
    point: GeoPoint
    timestamp: datetime
    elapsed_seconds: float = 0.0

@dataclass
class TrackAnalysis:
    """Track analysis results"""
    speed: float  # m/s
    acceleration: float  # m/s²
    distance_from_start: float  # meters

class Track:
    """Track with dynamic analysis"""
    def __init__(self, points: List[TrackPoint]):
        self.points = points
        self.analysis: List[TrackAnalysis] = []
        self._analyze()

    def _analyze(self):
        """Perform speed and acceleration analysis, ensuring strictly increasing times for spline."""
        if len(self.points) < 2:
            self.analysis = [TrackAnalysis(speed=0.0, acceleration=0.0, distance_from_start=0.0) for _ in self.points]
            return

        # 1. Calculate distances and speeds for ALL original points
        all_distances = [0.0]
        all_speeds = [0.0] # Speed at first point is 0
        
        for i in range(1, len(self.points)):
            dist = self.points[i-1].point.distance_to(self.points[i].point)
            dt = self.points[i].elapsed_seconds - self.points[i-1].elapsed_seconds
            all_distances.append(all_distances[-1] + dist)
            # Avoid division by zero, use previous speed if time hasn't advanced
            all_speeds.append(dist / dt if dt > 0 else (all_speeds[-1] if all_speeds else 0.0))

        # 2. Prepare data for Akima spline with strictly increasing times
        unique_times = []
        unique_speeds_for_spline = []
        
        if self.points:
            # Always include the first point
            unique_times.append(self.points[0].elapsed_seconds)
            unique_speeds_for_spline.append(all_speeds[0])

            for i in range(1, len(self.points)):
                current_time = self.points[i].elapsed_seconds
                current_speed = all_speeds[i]
                
                # Only add point if its time is strictly greater than the last unique time
                if current_time > unique_times[-1]:
                    unique_times.append(current_time)
                    unique_speeds_for_spline.append(current_speed)
                # If time is not strictly increasing, skip adding to unique_times, but still accumulate distances/speeds
                # for `all_distances` and `all_speeds`

        # 3. Calculate acceleration using Akima spline on unique time points
        all_accelerations = [0.0] * len(self.points)

        # Akima spline requires at least 4 unique points
        if len(unique_times) >= 4:
            try:
                # Use numpy arrays for interpolation for better performance
                np_unique_times = np.array(unique_times)
                np_unique_speeds = np.array(unique_speeds_for_spline)

                # Create Akima spline for speed
                spline = Akima1DInterpolator(np_unique_times, np_unique_speeds)
                
                # Interpolate acceleration (first derivative of speed) for all original points' times
                for i, p in enumerate(self.points):
                    # Ensure the point's time is within the range of the spline's x-values
                    if np_unique_times[0] <= p.elapsed_seconds <= np_unique_times[-1]:
                        # Evaluate the first derivative (acceleration) at this point's elapsed_seconds
                        all_accelerations[i] = spline(p.elapsed_seconds, 1)
                    # If outside the range, acceleration remains 0.0 (initialized value)
                    
            except Exception as e:
                print(f"Speed spline error: {e}")
        elif len(unique_times) >= 2: # Fallback to linear interpolation for acceleration if Akima not possible
            try:
                np_unique_times = np.array(unique_times)
                np_unique_speeds = np.array(unique_speeds_for_spline)
                linear_spline = interp1d(np_unique_times, np_unique_speeds, kind='linear', fill_value="extrapolate")
                # Approximate acceleration as change in speed / change in time
                for i, p in enumerate(self.points):
                    if i > 0 and np_unique_times[0] <= p.elapsed_seconds <= np_unique_times[-1]:
                        # Simple numerical derivative approximation
                        # Look for a small epsilon time forward/backward to estimate derivative
                        t_current = p.elapsed_seconds
                        # Ensure we have points to estimate derivative
                        if len(np_unique_times) > 1:
                            # Find the closest unique_time point for current_time
                            idx = np.searchsorted(np_unique_times, t_current)
                            if idx == 0: # At the very beginning
                                if len(np_unique_times) > 1:
                                    dt = np_unique_times[1] - np_unique_times[0]
                                    if dt > 0:
                                        all_accelerations[i] = (np_unique_speeds[1] - np_unique_speeds[0]) / dt
                            elif idx == len(np_unique_times): # At the very end
                                dt = np_unique_times[-1] - np_unique_times[-2]
                                if dt > 0:
                                    all_accelerations[i] = (np_unique_speeds[-1] - np_unique_speeds[-2]) / dt
                            else: # Somewhere in between, use average of left and right slopes or linear interpolation's derivative
                                # Simple central difference on interpolated speed
                                delta_t = 0.1 # Small time step for derivative approximation
                                speed_at_t_plus_dt = float(linear_spline(t_current + delta_t))
                                speed_at_t_minus_dt = float(linear_spline(t_current - delta_t))
                                all_accelerations[i] = (speed_at_t_plus_dt - speed_at_t_minus_dt) / (2 * delta_t)

            except Exception as e:
                print(f"Speed spline (linear fallback) error: {e}")

        # 4. Construct self.analysis for ALL original points
        self.analysis = [
            TrackAnalysis(
                speed=all_speeds[i],
                acceleration=all_accelerations[i],
                distance_from_start=all_distances[i]
            )
            for i in range(len(self.points))
        ]