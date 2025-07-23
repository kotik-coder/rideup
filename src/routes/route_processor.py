# route_processor.py
from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import Akima1DInterpolator, interp1d
from typing import Any, Dict, List, Tuple

from src.ui.map_helpers import geodesic_integrand, print_step
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
        """
        Creates a smoothed version of the route's points and elevations using
        linear or Akima spline interpolation based on point density.
        """
        MIN_POINTS_FOR_LINEAR   = 100  # Use linear if ≥ 100 points
        MAX_DISTANCE_FOR_LINEAR = 15.0 # Use linear if avg spacing <15m

        points = route.points

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

        try:
            lats = [p.lat for p in points]
            lons = [p.lon for p in points]
            
            if use_akima:
                interp_lat = Akima1DInterpolator(t, lats)
                interp_lon = Akima1DInterpolator(t, lons)
                interp_elev = Akima1DInterpolator(t, route.elevations)
            else:
                interp_lat = interp1d(t, lats, kind='linear')
                interp_lon = interp1d(t, lons, kind='linear')
                interp_elev = interp1d(t, route.elevations, kind='linear')

            self.smooth_points = [
                GeoPoint(lat, lon, elev)
                for lat, lon, elev in zip(
                    interp_lat(t_smooth),
                    interp_lon(t_smooth),
                    interp_elev(t_smooth)
                )
            ]

            self.interpolators= {'lat' : interp_lat,
                                 'lon' : interp_lon,
                                 'ele' : interp_elev }
        
        except Exception as e:
            print_step("Error", f"Smoothing failed for {route.name}: {str(e)}", level="ERROR")
            return route.points.copy(), None, None, None
        
    def calculate_precise_distances(self):
        """
        Calculates precise cumulative distances along the route using stored interpolators.
        Handles both Akima and linear interpolation methods.
        """
        if not self.smooth_points or not self.interpolators.get('lat') or not self.interpolators.get('lon'):
            print_step("DistanceCalc", "Missing required data for distance calculation", level="WARNING")
            return

        # Get derivatives based on interpolator type
        lat_interp = self.interpolators['lat']
        lon_interp = self.interpolators['lon']
        
        # Create proper derivative functions based on interpolator type
        def create_derivative(interp):
            if isinstance(interp, Akima1DInterpolator):
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
        processed_route = ProcessedRoute( route )                
        
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