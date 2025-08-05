from typing import Any, List, Dict
import numpy as np

from src.routes.profile_analyzer import SegmentProfile, StaticProfile
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

        return {
            'static': static_profile,
            'dynamic': associated_tracks[0].analysis,
            'segments': segment_profile,
            # 'difficulty': self._determine_difficulty(analysed_segments, baseline_features).name,
        }
