from typing import Any, List, Dict
import numpy as np

from src.routes.profile_analyzer import SegmentProfile, StaticProfile, StaticProfilePoint
from src.routes.route_processor import ProcessedRoute
from src.routes.track import Track
from src.ui.map_helpers import print_step
from scipy.signal import argrelextrema
from src.routes.trail_features import *

class StatisticsCollector:    
        
    def __init__(self):
        print_step("StatisticsCollector", "Initialized with enhanced baseline analysis")

    def generate_route_profiles(self, 
                             proute: ProcessedRoute,
                             associated_tracks: List[Track]) -> Dict[str, Any]:
        
        static_profile  = StaticProfile(proute)
        segment_profile = SegmentProfile(static_profile, proute)
        
        # Enhanced difficulty analysis with baseline features
        baseline_features = self._analyze_baseline_features(static_profile, proute)
        
        return {
            'static': static_profile,
            'dynamic': associated_tracks[0].analysis,
            'segments': segment_profile,
            # 'difficulty': self._determine_difficulty(analysed_segments, baseline_features).name,
            'baseline_features': baseline_features
        }
    
    def _analyze_baseline_features(self, profile : StaticProfile, proute: ProcessedRoute) -> Dict[str, Any]:
        """Analyze baseline features using interpolation"""
        distances = [p.distance_from_origin for p in profile.points]
        
        baseline_gradients = profile.get_baseline(proute.baseline, mode = 'gradients')
        
        # Find local maxima/minima in baseline gradient
        maxima = argrelextrema(baseline_gradients, np.greater)[0]
        minima = argrelextrema(baseline_gradients, np.less)[0]
        extrema = sorted(np.concatenate((maxima, minima)))
        
        # Analyze sustained climbs/descents
        sustained_segments = []
        for i in range(1, len(extrema)):
            start_idx = extrema[i-1]
            end_idx = extrema[i]
            length = distances[end_idx] - distances[start_idx]
            avg_gradient = np.mean(baseline_gradients[start_idx:end_idx+1])
            
            if abs(avg_gradient) > 0.01 and length > 50:
                seg_type = (GradientSegmentType.ASCENT if avg_gradient > 0 
                            else GradientSegmentType.DESCENT)
                sustained_segments.append({
                    'start': distances[start_idx],
                    'end': distances[end_idx],
                    'length': length,
                    'gradient': avg_gradient,
                    'type': seg_type.name
                })
        
        return {
            'sustained_segments': sustained_segments,
            'dominant_frequencies': proute.baseline.freqs.tolist() if hasattr(proute.baseline, 'freqs') else []
        }