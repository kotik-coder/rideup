# route_processor.py
from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator, interp1d, CubicSpline
from typing import Any, Dict, List, Tuple
from scipy.signal import savgol_filter

from src.routes.mutable_route import EstablishedRoute
from src.routes.route_helpers import calculate_sg_window_length, geodesic_integrand, resample_uniformly
from src.routes. baseline import Baseline
from src.routes.route import GeoPoint, Route
from src.routes.checkpoints import Checkpoint, CheckpointGenerator
from src.routes.track import Track
from src.iio.spot_photo import SpotPhoto
from src.ui.map_helpers import print_step

class ProcessedRoute:
    smooth_points: List[GeoPoint]
    checkpoints: List[Checkpoint]
    bounds: List[float]
    interpolators: Dict[str, Any]
    baseline : Baseline
        
    def __init__(self, route : Route):
        self._create_smooth_route(route)
        self.checkpoints = []
        self.bounds = []
        
    def get_oscillations(self, start_index, end_index):        
        ps = self.smooth_points
        dist = lambda i: ps[i].distance_from_origin/ps[-1].distance_from_origin
        eles = lambda i: ps[i].elevation
        base = lambda i: self.baseline.get_baseline_elevation(dist(i))
        
        return np.array([ eles(i) - base(i) for i in range(start_index, end_index)])
    
    def total_distance(self):
        return self.smooth_points[-1].distance_from_origin        
      
    def get_elevation(self, t: float) -> float:
        """Get interpolated elevation at normalized position t (0-1)"""
        return float(self.interpolators['ele'](t))
    
    def get_gradient(self, t: float) -> float:
        L = self.smooth_points[-1].distance_from_origin
        """Calculate gradient at normalized position t (0-1)"""
        if isinstance(self.interpolators['ele'], CubicSpline):
            return float(self.interpolators['ele'].derivative()(t)) / L
        else:
            # For linear interpolation, use finite differences
            epsilon = 0.001
            t1 = max(0, t - epsilon)
            t2 = min(1, t + epsilon)
            return (self.get_elevation(t2) - self.get_elevation(t1)) / (t2 - t1) / L
    
    def get_gradient_at_distance(self, distance: float) -> float:
        """Calculate gradient at specific distance along route"""
        t = distance / self.total_distance()
        return self.get_gradient(t)
    
    def find_closest_route_point(self, point: GeoPoint) -> Tuple[int, GeoPoint]:
        index = min(
            range(len(self.smooth_points)),
            key=lambda i: self.smooth_points[i].distance_to(point)
        )
        return index, self.smooth_points[index]    

    def _create_smooth_route(self, route: Route):
        """Creates a smoothed version of the route's points and elevations using
        Savitzky-Golay filtering for elevations, with PCHIP/Akima interpolation
        where appropriate, while ensuring baseline is calculated on uniform samples.
        """
        MIN_POINTS_FOR_LINEAR = 200    # Use linear if ≥ 200 points
        MAX_DISTANCE_FOR_LINEAR = 5.0  # Use linear if avg spacing <5.0m

        points = route.points
        
        # Calculate cumulative distances
        distances = np.zeros(len(points))
        for i in range(1, len(points)):
            distances[i] = distances[i-1] + points[i-1].distance_to(points[i])
        
        avg_distance = distances[-1] / (len(points) - 1)
        
        use_splines = (len(points) < MIN_POINTS_FOR_LINEAR) or \
                      (avg_distance > MAX_DISTANCE_FOR_LINEAR)
        
        # First create uniformly resampled version for baseline calculation
        uniform_distances, uniform_elevations, t_uniform = resample_uniformly(distances, 
                                                                              route.elevations, 
                                                                              MIN_POINTS_FOR_LINEAR, 
                                                                              MAX_DISTANCE_FOR_LINEAR)
        
        # Calculate baseline on uniform samples
        #baseline_elev, dominant_freqs = calculate_baseline_precise(uniform_elevations, uniform_distances)
        self.baseline = Baseline(uniform_elevations, uniform_distances)
        
        # Now process original points with selected interpolation method
        t = distances / distances[-1]
        lats = [p.lat for p in points]
        lons = [p.lon for p in points]
        
        # Calculate residuals on original points
        interp_baseline = interp1d(
            t_uniform, 
            self.baseline.y,
            kind='linear', 
            fill_value='extrapolate'
        )
        
        baseline_on_original = interp_baseline(t)
        oscillations = route.elevations - baseline_on_original        
        
        # Smooth oscillations
        try:
            smoothed_oscillations = savgol_filter(
                oscillations, 
                window_length=calculate_sg_window_length(self.baseline.freqs, distances, oscillations), 
                polyorder=3
            )
        except Exception as e:
            from scipy.ndimage import uniform_filter1d
            smoothed_oscillations = uniform_filter1d(oscillations, size=3)
        
        # Combine components
        smoothed_elev = baseline_on_original + smoothed_oscillations
        
        # Create final interpolators (using original interpolation method choice)
        if use_splines:
            interp_lat = PchipInterpolator(t, lats)
            interp_lon = PchipInterpolator(t, lons)
            interp_ele = PchipInterpolator(t, smoothed_elev)
            interp_baseline_final = PchipInterpolator(t, baseline_on_original)
        else:
            interp_lat = interp1d(t, lats, kind='linear')
            interp_lon = interp1d(t, lons, kind='linear')
            interp_ele = interp1d(t, smoothed_elev, kind='linear')
            interp_baseline_final = interp1d(t, baseline_on_original, kind='linear')
        
        # Generate smooth points (using original method's point count logic)
        num_output_points = int(max(MIN_POINTS_FOR_LINEAR, distances[-1] / MAX_DISTANCE_FOR_LINEAR))
        t_smooth = np.linspace(0, 1, num_output_points)
        
        self.smooth_points = [
            GeoPoint(lat, lon, elev)
            for lat, lon, elev in zip(
                interp_lat(t_smooth),
                interp_lon(t_smooth),
                interp_ele(t_smooth)
            )
        ]
        
        self.baseline.interpolation = interp_baseline_final
        
        self.interpolators = {
            'lat': interp_lat,
            'lon': interp_lon,
            'ele': interp_ele,
            'baseline': interp_baseline_final
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

# route_processor.py (add this method to the RouteProcessor class)
class RouteProcessor:
    def __init__(self, local_photos: List[SpotPhoto], all_tracks: List[Track]):
        self.checkpoint_generator = CheckpointGenerator(local_photos)
        self.all_tracks = all_tracks

    def process_route(self, route: Route) -> ProcessedRoute:
        """Main processing pipeline for a route - works for both Route and EstablishedRoute"""
        if not route:
            return        
        
        print_step("RouteProcessor", f"Starting processing for route: {route.name}")        

        # Get tracks associated with this route
        associated_tracks = []
        if isinstance(route, EstablishedRoute):
            # For established routes, get tracks from all original routes
            for original_route in route.original_routes.values():
                route_tracks = [t for t in self.all_tracks if t.route and t.route.name == original_route.name]
                associated_tracks.extend(route_tracks)
        else:
            # For regular routes, get tracks for this specific route
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