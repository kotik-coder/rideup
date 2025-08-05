from typing import Any, List, Optional
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize_scalar
import numpy as np
from scipy.interpolate import CubicSpline

class Baseline:
    
    x : np.ndarray
    y : np.ndarray
    freqs : Optional[np.ndarray]
    interpolation : Any
    
    def __init__(self, elevations, distances, method: str = 'fast'):
        """
        Initialize baseline calculator.
        
        Parameters:
        - elevations: Array of elevation values
        - distances: Array of distance values
        - precise: Whether to use precise mode (default True)
        - method: Baseline calculation method ('auto', 'als', or 'chebyshev')
        """
        self.x = distances
        self.method = method.lower()
        
        if self.method == 'precise':
            self.y, self.freqs = self._calculate_baseline_precise(elevations, distances)
        elif self.method == 'fast':
            self.y = self._als_baseline(elevations, lam=1500, p=0.01)    
            self.freqs = self._elevation_frequencies(elevations, distances, 0.0, 1e6)        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'fast' or 'precise'")
        
    def _create_interpolation(self):
        """Create interpolation function based on normalized x coordinates (0-1)"""
        x_norm = (self.x - self.x[0]) / (self.x[-1] - self.x[0])
        self.interpolation = CubicSpline(x_norm, self.y)
            
    def get_baseline_elevation(self, t: float) -> float:
        """Get baseline elevation at normalized position t (0-1)"""
        return float(self.interpolation(t))

    def get_baseline_gradient(self, t: float) -> float:        
        """Calculate baseline gradient at normalized position t (0-1)"""
        if isinstance(self.interpolation, CubicSpline):
            return float(self.interpolation.derivative()(t)) / (self.x[-1] - self.x[0])
        else:
            # For linear interpolation, use finite differences
            epsilon = 0.001
            t1 = max(0, t - epsilon)
            t2 = min(1, t + epsilon)
            return (self.get_baseline_elevation(t2) - self.get_baseline_elevation(t1)) / (t2 - t1) / (self.x[-1] - self.x[0])
    
    # ALS smoothing with endpoint constraints
    def _als_baseline(self, y, lam=100, p=0.01, n_iter=10) -> np.ndarray:
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
            w = 0.8*p * (y > baseline) + 0.2*(1 - p) * (y <= baseline)  # Changed weights
            
            # Re-enforce endpoint weights
            w[0] = w[-1] = 1e9
            
        return baseline    

    # Step 3: Find optimal lambda in this range
    def _optimize_als_params(self, elevations, lam_range=(1, 1e6), p_range=(1E-6, 0.1)):

        # Cache to avoid redundant computations
        param_cache = {}
        
        def evaluate_params(lam, p):
            """Modified evaluation function to prioritize low frequencies"""
            key = (round(lam,6), round(p,6))
            if key not in param_cache:
                bl = self._als_baseline(elevations, lam=lam, p=p)
                # Higher weight on smoothness (2nd derivative)
                residual = np.median(np.abs(bl - elevations))  # Robust fit
                smoothness = np.sum(np.diff(bl, 2)**2)       # 2nd derivative roughness
                param_cache[key] = 0.8*smoothness + 0.2*residual  # Changed weights
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

    def _calculate_baseline_als(self, elevations, distances):
        """Calculate baseline using ALS method (existing precise implementation)"""
        total_distance = distances[-1]
        dominant_freqs = self._elevation_frequencies(elevations, distances, 0.001, 1e6)
        cutoff_freq = min(dominant_freqs.max(), 1e6) if len(dominant_freqs) > 0 else 1e6
        
        initial_lam = 10 ** (4 - (cutoff_freq/0.001)) if cutoff_freq > 0 else 1000
        lam_range = (max(1, initial_lam / 10), min(1e6, initial_lam * 10))
        optimal_lam, optimal_p = self._optimize_als_params(elevations, lam_range)
        baseline = self._als_baseline(elevations, lam=optimal_lam, p=optimal_p)
        
        return baseline, dominant_freqs
    
    def _calculate_baseline_precise(self, elevations, distances):
        total_distance = distances[-1]
            
        # First perform FFT analysis to get frequency info 
        sampling_freq = len(distances) / total_distance
        nyquist = sampling_freq / 2
        #bounds for peak detection in FFT power spectrum
        max_freq = nyquist
        min_freq = max(1/total_distance, 0.001)

        dominant_freqs = self._elevation_frequencies(elevations, distances, min_freq, max_freq)
        cutoff_freq    = min(dominant_freqs.max(), max_freq) if len(dominant_freqs) > 0 else max_freq

        # Step 1: Get initial lambda estimate from FFT
        initial_lam = 10 ** (4 - (cutoff_freq/min_freq)) if cutoff_freq > 0 else 1000
        
        # Step 2: Define optimization neighborhood around initial estimate
        lam_range = (
            max(1,   initial_lam / 10),  # Lower bound
            min(1e6, initial_lam * 10)   # Upper bound
        )
        
        optimal_lam, optimal_p = self._optimize_als_params(elevations, lam_range)
        
        # Final baseline with optimized lambda
        baseline = self._als_baseline(elevations, lam=optimal_lam, p=optimal_p)
        
        return baseline, dominant_freqs
    
    def _elevation_frequencies(self, elevations, distances, min_freq, max_freq):
        total_distance = distances[-1]
        sampling_freq = len(distances) / total_distance

        n_fft = len(elevations)
        fft_vals = np.fft.fft(elevations, n=n_fft)
        freqs    = np.fft.fftfreq(n_fft, d=1/sampling_freq)
        
        # Find dominant frequencies (same as before)
        pos_mask        = freqs > 0
        psd             = np.abs(fft_vals[pos_mask])**2
        psd_freqs       = freqs[pos_mask]
        smooth_psd      = uniform_filter1d(psd, size=max(3, int(len(psd)/10)))
        valid_mask      = (psd_freqs >= min_freq) & (psd_freqs <= max_freq)
        candidate_freqs = psd_freqs[valid_mask]
        candidate_psd   = smooth_psd[valid_mask]
        
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