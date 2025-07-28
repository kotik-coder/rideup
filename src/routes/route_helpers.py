import numpy as np
import math
from scipy.interpolate import interp1d


# Define WGS-84 ellipsoid constants for high accuracy
WGS84_A = 6378137.0  # Semi-major axis in meters
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_E2 = WGS84_F * (2 - WGS84_F)  # Square of first eccentricity

TILE_SIZE = 512 # Pixel size of a single map tile. Common values are 256 or 512.
EARTH_CIRCUMFERENCE = 40075016.686 # In meters
DEGREES_PER_RADIAN = 180 / math.pi

def calculate_sg_window_length(dominant_freqs, distances, oscillations):
    if len(dominant_freqs) > 0:
        # Use the highest physically meaningful frequency
        max_meaningful_freq = dominant_freqs[-1]  
        min_wavelength = 1/max_meaningful_freq  # In meters
        
        # Convert wavelength to points
        avg_point_spacing = distances[-1] / len(oscillations)
        window_points = int(min_wavelength / avg_point_spacing)
        
        # Ensure odd and within bounds
        window_length = max(5, min(window_points, len(oscillations)))
        window_length = window_length + 1 if window_length % 2 == 0 else window_length
    else:
        # Fallback to adaptive default
        window_length = min(11, len(oscillations))
        
        return window_length

def resample_uniformly(x, y, min_points = 100, max_distance = 15.0):
        
    # First create uniformly resampled version for baseline calculation
    num_uniform_points = int(max(min_points, x[-1] / max_distance))
    t_uniform = np.linspace(0, 1, num_uniform_points)
    
    # Linear interpolation for uniform resampling (for baseline calculation only)
    lin_interp = interp1d( x/x[-1], y, kind='linear' )
    uniform_y = lin_interp(t_uniform)
    uniform_x = t_uniform * x[-1]
    
    return uniform_x, uniform_y, t_uniform

def verify_uniform_sampling(distances):
    # --- Strict Uniform Sampling Check ---
    distance_deltas = np.diff(distances)
    uniform_threshold = 0.01  # 1% variation allowed
    
    if len(distance_deltas) > 0:  # Only check if we have deltas
        mean_delta = distance_deltas.mean()
        max_deviation = np.abs(distance_deltas - mean_delta).max()
        
        if max_deviation / mean_delta > uniform_threshold:
            raise ValueError(
                f"Input distances must be uniformly sampled. Found spacing varying by "
                f"{max_deviation/mean_delta:.1%} (allowed: <{uniform_threshold:.0%}).\n"
                f"Mean spacing: {mean_delta:.2f}m, Range: {distance_deltas.min():.2f}m "
                f"to {distance_deltas.max():.2f}m"
            )   

def geodesic_integrand(t, interp_lat, interp_lon, dlat_dt, dlon_dt):#
    """
    The function to be integrated to find the arc length.
    This corresponds to the ds/dt term in the mathematical explanation.
    """
    # Get latitude and its derivative at parameter t
    lat_deg = interp_lat(t)
    dlat_dt_deg = dlat_dt(t)
    # Get longitude derivative at parameter t
    dlon_dt_deg = dlon_dt(t)

    # Convert degrees (from spline) to radians for trigonometric functions
    lat_rad = np.radians(lat_deg)
    dlat_dt_rad = np.radians(dlat_dt_deg)
    dlon_dt_rad = np.radians(dlon_dt_deg)

    # Calculate radii of curvature M and N
    sin_lat_sq = np.sin(lat_rad)**2
    den = np.sqrt(1 - WGS84_E2 * sin_lat_sq)
    
    M = (WGS84_A * (1 - WGS84_E2)) / (den**3)  # Meridional radius
    N = WGS84_A / den  # Prime vertical radius
    
    # Calculate the term under the square root
    integrand_val_sq = (M * dlat_dt_rad)**2 + (N * np.cos(lat_rad) * dlon_dt_rad)**2
    
    return np.sqrt(integrand_val_sq)