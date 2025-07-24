# route_processor.py
from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator, interp1d
from typing import Any, Dict, List, Tuple
from scipy.signal import savgol_filter

from src.ui.map_helpers import fft_lowpass_filter, geodesic_integrand, print_step
from src.routes.route import GeoPoint, Route
from src.routes.checkpoints import Checkpoint, CheckpointGenerator
from src.routes.track import Track
from src.iio.spot_photo import SpotPhoto

class ProcessedRoute:
    smooth_points: List[GeoPoint]
    checkpoints: List[Checkpoint]
    bounds: List[float]
    interpolators: Dict[str, Any]
        
    def __init__(self, route : Route):
        self._create_smooth_route(route)
        self.checkpoints = []
        self.bounds = []    
    
    def find_closest_route_point(self, point: GeoPoint) -> Tuple[int, GeoPoint]:
        index = min(
            range(len(self.smooth_points)),
            key=lambda i: self.smooth_points[i].distance_to(point)
        )
        return index, self.smooth_points[index]

    def _create_smooth_route(self, route: Route):
        """Creates a smoothed version of the route's points and elevations using
        Savitzky-Golay filtering for elevations, and linear/Akima spline 
        interpolation based on point density. Also calculates FFT baseline.
        """
        MIN_POINTS_FOR_LINEAR = 100  # Use linear if ≥ 100 points
        MAX_DISTANCE_FOR_LINEAR = 15.0  # Use linear if avg spacing <15m

        points = route.points

        # Calculate cumulative distances
        distances = np.zeros(len(points))
        for i in range(1, len(points)):
            distances[i] = distances[i-1] + points[i-1].distance_to(points[i])
        
        avg_distance = distances[-1] / (len(points) - 1)
        
        use_akima = (len(points) < MIN_POINTS_FOR_LINEAR) or \
                (avg_distance > MAX_DISTANCE_FOR_LINEAR)
        
        method = "Akima" if use_akima else "linear"
        print_step("Smoothing", 
                f"Route '{route.name}': {len(points)} points, "
                f"avg spacing {avg_distance:.1f}m → {method} interpolation")

        t = distances / distances[-1]
        num_smooth_points = int(max(MIN_POINTS_FOR_LINEAR, 
                                distances[-1] / MAX_DISTANCE_FOR_LINEAR))
        t_smooth = np.linspace(0, 1, num_smooth_points)

        lats = [p.lat for p in points]
        lons = [p.lon for p in points]
        
        # Apply FFT low-pass filter to get baseline elevations
        baseline_elev, cutoff_freq, dominant_freqs = fft_lowpass_filter(
            route.elevations, distances
        )
        
        print_step("FFT Analysis", 
                f"Cutoff frequency: {cutoff_freq:.4f} cycles/meter "
                f"(wavelength: {1/cutoff_freq:.1f}m)\n"
                f"Dominant frequencies: {dominant_freqs[:3]} cycles/meter")
        
        # Calculate residual oscillations (original - baseline)
        oscillations = route.elevations - baseline_elev
        
        # Apply Savitzky-Golay smoothing to residual oscillations
        window_length = 10
        polyorder = min(3, window_length - 1)  # Ensure polyorder < window_length
        
        try:
            smoothed_oscillations = savgol_filter(oscillations, window_length, polyorder)
        except Exception as e:
            # Fallback to simple moving average if Savitzky-Golay fails
            from scipy.ndimage import uniform_filter1d
            window_size = max(3, window_length // 3)
            smoothed_oscillations = uniform_filter1d(oscillations, size=window_size)
            print_step("Warning", f"Savitzky-Golay failed, using moving average: {str(e)}", level="WARNING")
        
        # Combine baseline with smoothed oscillations
        smoothed_elev = baseline_elev + smoothed_oscillations
        
        if use_akima:
            interp_lat = PchipInterpolator(t, lats)
            interp_lon = PchipInterpolator(t, lons)
            interp_elev = PchipInterpolator(t, smoothed_elev)
            interp_baseline = PchipInterpolator(t, baseline_elev)
        else:
            interp_lat = interp1d(t, lats, kind='linear')
            interp_lon = interp1d(t, lons, kind='linear')
            interp_elev = interp1d(t, smoothed_elev, kind='linear')
            interp_baseline = interp1d(t, baseline_elev, kind='linear')

        self.smooth_points = [
            GeoPoint(lat, lon, elev)
            for lat, lon, elev in zip(
                interp_lat(t_smooth),
                interp_lon(t_smooth),
                interp_elev(t_smooth)
            )
        ]

        self.interpolators = {
            'lat': interp_lat,
            'lon': interp_lon,
            'ele': interp_elev,
            'baseline': interp_baseline  # Store baseline interpolator
        }        
        
    def calculate_precise_distances(self):
        """
        Calculates precise cumulative distances along the route using stored interpolators.
        Handles both PChip and linear interpolation methods.
        """
        if not self.smooth_points or not self.interpolators.get('lat') or not self.interpolators.get('lon'):
            print_step("DistanceCalc", "Missing required data for distance calculation", level="WARNING")
            return

        # Get derivatives based on interpolator type
        lat_interp = self.interpolators['lat']
        lon_interp = self.interpolators['lon']
        
        # Create proper derivative functions based on interpolator type
        def create_derivative(interp):
            if isinstance(interp, PchipInterpolator):
                return interp.derivative()
            else:
                # For linear interpolation, we need to create a proper gradient function
                t_values = np.linspace(0, 1, len(self.smooth_points))
                lat_values = interp(t_values)
                
                # Pre-compute gradients at all points
                gradients = np.gradient(lat_values, t_values)
                
                # Return a function that interpolates these gradients
                return interp1d(t_values, gradients, kind='linear', fill_value='extrapolate')

        dlat_dt = create_derivative(lat_interp)
        dlon_dt = create_derivative(lon_interp)

        # Initialize distances
        self.smooth_points[0].distance_from_origin = 0.0
        cumulative_distance = 0.0

        t_values = np.linspace(0, 1, len(self.smooth_points))
        t_prev = t_values[0]  # Initialize with first parameter value

        # Calculate distances for each point
        for i in range(1, len(self.smooth_points)):
            t_curr = t_values[i]
            
            # Integrate over the segment
            segment_distance, _ = quad(
                geodesic_integrand,
                t_prev,
                t_curr,
                args=(self.interpolators['lat'], self.interpolators['lon'], dlat_dt, dlon_dt)
            )
            
            cumulative_distance += segment_distance
            self.smooth_points[i].distance_from_origin = cumulative_distance
            t_prev = t_curr  # Update for next iteration

class RouteProcessor:
    def __init__(self, local_photos: List[SpotPhoto], all_tracks: List[Track]):
        self.checkpoint_generator = CheckpointGenerator(local_photos)
        self.all_tracks = all_tracks

    def process_route(self, route: Route) -> ProcessedRoute:
        """Main processing pipeline for a route."""
    
        if not route:
            return        
        
        print_step("RouteProcessor", f"Starting processing for route: {route.name}")        

        # Get tracks associated with this route
        associated_tracks = [t for t in self.all_tracks if t.route == route]        
        processed_route = ProcessedRoute(route)                
        
        processed_route.checkpoints = self.checkpoint_generator.generate_checkpoints(
            processed_route.smooth_points,
            associated_tracks
        )
        print_step("RouteProcessor", f"Route '{route.name}': Generated {len(processed_route.checkpoints)} checkpoints.")
                
        processed_route.calculate_precise_distances()
        
        lons = [p.lon for p in route.points]
        lats = [p.lat for p in route.points]
        
        processed_route.bounds = [min(lons), min(lats), max(lons), max(lats)]                

        return processed_route