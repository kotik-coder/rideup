from typing import Any, List, Dict
import numpy as np

from src.routes.spot import Spot
from src.routes.profile_analyzer import SegmentProfile, StaticProfile
from src.routes.route_processor import ProcessedRoute
from src.routes.track import Track
from src.routes.trail_features import *

def generate_route_profiles(spot : Spot, proute: ProcessedRoute, associated_tracks: List[Track]):
    
    static_profile = StaticProfile(spot.system, proute)  # Pass system
    segment_profile = SegmentProfile(static_profile, proute)

    return {
        'static': static_profile,
        'dynamic': associated_tracks[0].analysis,
        'segments': segment_profile,
        # 'difficulty': self._determine_difficulty(analysed_segments, baseline_features).name,
    }