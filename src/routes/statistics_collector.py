from typing import Any, List, Dict
import numpy as np

from src.routes.spot import Spot
from src.routes.profile_analyzer import SegmentProfile, StaticProfile
from src.routes.route_processor import ProcessedRoute
from src.routes.track import Track
from src.routes.trail_features import *

def generate_route_profiles(spot: Spot, proute: ProcessedRoute, associated_tracks: List[Track]):
    static_profile = StaticProfile(spot.system, proute)
    segment_profile = SegmentProfile(static_profile, proute)
    difficulty = _determine_difficulty(segment_profile, static_profile, spot.system)
    
    return {
        'static': static_profile,
        'dynamic': associated_tracks[0].analysis,
        'segments': segment_profile,
        'difficulty': difficulty
    }

def _determine_difficulty(segment_profile: SegmentProfile, static_profile: StaticProfile, spot_system: RatingSystem) -> str:
    """
    Calculate route difficulty based on multiple factors with weighted scoring
    """
    if not segment_profile.segments:
        return "GREEN"
    
    # Initialize scoring components
    elevation_score = 0
    steepness_score = 0
    technical_score = 0
    feature_score = 0
    
    # Calculate elevation components
    total_distance = segment_profile.segments[-1].distances[-1]
    elev_gain = max(p.elevation for p in static_profile.points) - min(p.elevation for p in static_profile.points)
    
    # Normalize elevation gain (100m per km is considered challenging)
    elevation_score = min((elev_gain / total_distance) * 1000, 10)  # Max 10 points
    
    # Analyze segments
    for seg in segment_profile.segments:
        seg_length = seg.length()
        avg_grad = seg.avg_gradient()
        grad_percent = avg_grad * 100
        
        # Steepness scoring
        if seg.gradient_type == GradientSegmentType.STEEP_ASCENT:
            steepness_score += min(grad_percent / 5 * (seg_length/100), 5)  # Max 5 points per segment
        elif seg.gradient_type == GradientSegmentType.STEEP_DESCENT:
            steepness_score += min(abs(grad_percent) / 5 * (seg_length/100), 5)
        elif seg.gradient_type == GradientSegmentType.ASCENT:
            steepness_score += min(grad_percent / 10 * (seg_length/100), 2)
        elif seg.gradient_type == GradientSegmentType.DESCENT:
            steepness_score += min(abs(grad_percent) / 10 * (seg_length/100), 2)
        
        # Technical feature scoring
        if seg.feature_type == TrailFeatureType.TECHNICAL_ASCENT:
            technical_score += 3 * (seg_length/100)
        elif seg.feature_type == TrailFeatureType.TECHNICAL_DESCENT:
            technical_score += 4 * (seg_length/100)
        elif seg.feature_type in [TrailFeatureType.KICKER, TrailFeatureType.DROP]:
            technical_score += 2
        elif seg.feature_type == TrailFeatureType.SWITCHBACK:
            technical_score += 1
    
    # Calculate total score (weighted components)
    total_score = (
        0.3 * elevation_score + 
        0.4 * steepness_score + 
        0.3 * technical_score
    )
    
    # Normalize by distance (score per km)
    normalized_score = total_score / (total_distance / 1000)

    # Classify using spot system thresholds
    return RouteDifficulty.from_score(normalized_score, spot_system).name