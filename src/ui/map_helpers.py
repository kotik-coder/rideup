import math
from pathlib import Path
from typing import List
from datetime import datetime

from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import golden
from scipy.optimize import minimize_scalar

import numpy as np 

CACHE_FILE = Path("elevation_cache.json")
REQUEST_DELAY = 1.0  # Задержка между запросами в секундах

DEBUG = True

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
        print(window_length)
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
            
def identify_dominant_freqs(freqs, fft_vals, min_freq, max_freq):
    # Find dominant frequencies (same as before)
    pos_mask = freqs > 0
    psd = np.abs(fft_vals[pos_mask])**2
    psd_freqs = freqs[pos_mask]
    smooth_psd = uniform_filter1d(psd, size=max(3, int(len(psd)/10)))
    valid_mask = (psd_freqs >= min_freq) & (psd_freqs <= max_freq)
    candidate_freqs = psd_freqs[valid_mask]
    candidate_psd = smooth_psd[valid_mask]
    
    if len(candidate_psd) > 3:
        try:
            peaks, _ = find_peaks(candidate_psd,
                                height=np.median(candidate_psd),
                                distance=max(1, len(candidate_psd)//10))
            dominant_freqs = candidate_freqs[peaks]
        except ValueError:
            dominant_freqs = np.array([])
    else:
        dominant_freqs = np.array([])
        
    return dominant_freqs

# ALS smoothing with endpoint constraints
def als_baseline(y, lam=100, p=0.01, n_iter=10):
    """Asymmetric Least Squares baseline"""
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L-2, L))
    w = np.ones(L)
    
    # Force endpoints by giving them huge weights
    w[0] = w[-1] = 1e9
    
    for _ in range(n_iter):
        W = diags(w)
        Z = W + lam * D.T @ D
        baseline = spsolve(Z, w * y)
        w = p * (y > baseline) + (1 - p) * (y <= baseline)
        
        # Re-enforce endpoint weights
        w[0] = w[-1] = 1e9
        
    return baseline

# Step 3: Find optimal lambda in this range
def optimize_als_params(elevations, lam_range=(1, 1e6), p_range=(0.001, 0.1)):

    # Cache to avoid redundant computations
    param_cache = {}
    
    def evaluate_params(lam, p):
        """Cached evaluation of parameter set"""
        key = (round(lam,6), round(p,6))
        if key not in param_cache:
            bl = als_baseline(elevations, lam=lam, p=p)
            # Combined quality metric (lower is better)
            residual = np.median(np.abs(bl - elevations))  # Robust fit
            smoothness = np.sum(np.diff(bl, 2)**2)       # 2nd derivative roughness
            param_cache[key] = 0.7*smoothness + 0.3*residual
        return param_cache[key]
    
    # Stage 1: Coarse grid search
    lam_grid = np.logspace(np.log10(lam_range[0]), np.log10(lam_range[1]), 10)
    p_grid = np.linspace(p_range[0], p_range[1], 5)
    
    best_score = float('inf')
    best_lam, best_p = lam_range[0], p_range[0]
    
    for lam in lam_grid:
        for p in p_grid:
            current_score = evaluate_params(lam, p)
            if current_score < best_score:
                best_score = current_score
                best_lam, best_p = lam, p
    
    # Stage 2: Fine optimization around best coarse result
    def optimize_p(lam):
        """Optimize p for a given lambda"""
        res = minimize_scalar(
            lambda p: evaluate_params(lam, p),
            bounds=p_range,
            method='bounded'
        )
        return res.x, res.fun
    
    # Optimize lambda with nested p optimization
    final_result = minimize_scalar(
        lambda lam: optimize_p(lam)[1],
        bounds=(best_lam/10, best_lam*10),
        method='bounded'
    )
    optimal_lam = final_result.x
    optimal_p = optimize_p(optimal_lam)[0]
    
    return optimal_lam, optimal_p

def calculate_baseline(elevations, distances):
    """
    Enhanced baseline detection using:
    - FFT for frequency analysis
    - Asymmetric Least Squares (ALS) for constrained smoothing
    - Physical frequency constraints
    """
        
    # Input validation
    if len(elevations) < 4 or len(distances) < 4:
        return np.array(elevations), 0.01, np.array([0.01])

    total_distance = distances[-1]
    
    verify_uniform_sampling(distances)

    # First perform FFT analysis to get frequency info 
    sampling_freq = len(distances) / total_distance
    nyquist = sampling_freq / 2
    #bounds for peak detection in FFT power spectrum
    max_freq = nyquist/2
    min_freq = max(1/total_distance, 0.001)

    n_fft = len(elevations)
    fft_vals = np.fft.fft(elevations, n=n_fft)
    freqs    = np.fft.fftfreq(n_fft, d=1/sampling_freq)

    dominant_freqs = identify_dominant_freqs(freqs, fft_vals, min_freq, max_freq)    
    cutoff_freq    = min(dominant_freqs.max(), max_freq) if len(dominant_freqs) > 0 else max_freq

    # Step 1: Get initial lambda estimate from FFT
    initial_lam = 10 ** (4 - (cutoff_freq/min_freq)) if cutoff_freq > 0 else 1000
    
    # Step 2: Define optimization neighborhood around initial estimate
    lam_range = (
        max(1,   initial_lam / 10),  # Lower bound
        min(1e6, initial_lam * 10)   # Upper bound
    )
    
    optimal_lam, optimal_p = optimize_als_params(elevations, lam_range)
    
    # Final baseline with optimized lambda
    baseline = als_baseline(elevations, lam=optimal_lam, p=optimal_p)
    
    return baseline, dominant_freqs

def geodesic_integrand(t, interp_lat, interp_lon, dlat_dt, dlon_dt):
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

def print_step(prefix : str, message: str, level: str = "INFO"):    
    if(DEBUG):
        timestamp = datetime.now().strftime('%H:%M:%S')
        msg = f"[{timestamp}] [{prefix}] [{level}] {message}"
        if(str == 'ERROR'): 
            from colorama import Fore, Style            
            msg = Fore.RED + msg + Style.RESET_ALL
        print(msg)

def expanded_bounds(bounds : List[float], distance_km) -> List[float]: 
    lon_min, lat_min, lon_max, lat_max = bounds
    
    km_per_deg_lat = 111.0
        
    # For longitude, it depends on latitude
    avg_lat = (lat_min + lat_max) / 2
    km_per_deg_lon = 111.0 * math.cos(math.radians(avg_lat))
    
    # Calculate degree offsets
    lat_offset = distance_km / km_per_deg_lat
    lon_offset = distance_km / km_per_deg_lon
    
    return [
        lon_min - lon_offset,
        lat_min - lat_offset,
        lon_max + lon_offset,
        lat_max + lat_offset
    ]
    

def bounds_to_zoom(bounds, map_width_px, map_height_px, padding=0.0):
    """
    Calculates the optimal Mapbox zoom level to fit a given geographical bounding box.

    Args:
        bounds (dict): A dictionary with 'min_lon', 'max_lon', 'min_lat', 'max_lat'.
        map_width_px (int): The width of the map figure in pixels.
        map_height_px (int): The height of the map figure in pixels.
        padding (float): A factor to zoom out slightly for better framing (e.g., 0.1 for 10% padding).

    Returns:
        float: The calculated zoom level.
    """
    # 1. Calculate the required zoom level for the longitude span (width)
    lon_span = abs(bounds['max_lon'] - bounds['min_lon'])
    
    # Handle the case where the longitude span is zero
    if lon_span == 0:
      # If there's only one point, we can't calculate a zoom level from a span.
      # Default to a reasonable zoom level, or handle as needed.
      # Here, we'll just focus on latitude calculation.
      zoom_lon = 20 # A high zoom level for a single point
    else:
      zoom_lon = math.log2(map_width_px * 360 / (lon_span * TILE_SIZE))

    # 2. Calculate the required zoom level for the latitude span (height)
    # This is more complex due to the Mercator projection's distortion.
    def lat_to_mercator_y(lat):
        # Converts latitude to a 'y' coordinate in the Mercator projection
        lat_rad = math.radians(lat)
        return math.log(math.tan(math.pi / 4 + lat_rad / 2))

    y_merc_min = lat_to_mercator_y(bounds['min_lat'])
    y_merc_max = lat_to_mercator_y(bounds['max_lat'])
    lat_merc_span = abs(y_merc_max - y_merc_min)

    # Handle the case where the latitude span is zero
    if lat_merc_span == 0:
        zoom_lat = 20 # A high zoom level for a single point
    else:
        zoom_lat = math.log2(map_height_px / (lat_merc_span / (2 * math.pi) * TILE_SIZE))

    # 3. Choose the smaller of the two zoom levels to ensure the entire box fits
    zoom = min(zoom_lon, zoom_lat)
    
    # 4. Apply padding by slightly zooming out
    final_zoom = zoom * (1 - padding)

    return final_zoom