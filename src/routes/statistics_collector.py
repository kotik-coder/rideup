from typing import Any, List, Dict, Optional
import numpy as np

from src.routes.spot import Spot
from src.routes.profile_analyzer import Profile, ProfileSegment
from src.routes.route_processor import ProcessedRoute
from src.routes.track import Track
from src.routes.rating_system import *
from src.routes.terrain_and_weather import TerrainAnalysis, WeatherData

def generate_route_profiles(spot: Spot, proute: ProcessedRoute, associated_tracks: List[Track]) -> Dict[str, Any]:
    """Generate complete route analysis including profile and statistics"""
    profile = Profile(spot.system, proute)
    
    # Calculate additional statistics
    difficulty = _determine_difficulty(profile, spot.system)    
    
    return {
        'profile': profile,
        'difficulty': difficulty,
    }

def _determine_difficulty(profile: Profile, spot_system: RatingSystem) -> str:
    """
    Calculate route difficulty based on multiple factors with weighted scoring
    """
    if not profile.segments:
        return "GREEN"
    
    # Initialize scoring components
    elevation_score = 0
    steepness_score = 0
    technical_score = 0
    feature_score = 0
    
    # Calculate elevation components
    total_distance = profile.points[-1].distance_from_origin
    elev_gain = max(p.elevation for p in profile.points) - min(p.elevation for p in profile.points)
    
    # Normalize elevation gain (100m per km is considered challenging)
    elevation_score = min((elev_gain / total_distance) * 1000, 10)  # Max 10 points
    
    # Analyze segments
    for seg in profile.segments:
        seg_length = seg.length(profile.points)
        avg_grad = seg.grade(profile.points)
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
        
        # Enhanced feature scoring using RatingSystem parameters
        if seg.feature:
            feature_config = seg.feature.feature_type.get_config(spot_system)
            feature_score += feature_config.get('difficulty_impact', 0) * (seg_length/100)
            
        # Score short features
        for sf in seg.short_features:
            sf_config = sf.feature_type.get_config(spot_system)
            feature_score += sf_config.get('difficulty_impact', 0) * (sf.length(seg, profile.points)/10)
    
    # Calculate total score (weighted components)
    total_score = (
        0.2 * elevation_score + 
        0.3 * steepness_score + 
        0.5 * feature_score  
    )
    
    # Normalize by distance (score per km)
    normalized_score = total_score / (total_distance / 1000)

    # Classify using spot system thresholds
    return RouteDifficulty.from_score(normalized_score, spot_system).name