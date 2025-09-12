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
    Calculate route difficulty as weighted average of segment scores
    Score is independent of route length - it's an average per unit distance
    """
    if not profile.segments or not profile.points:
        return "GREEN"
    
    total_distance = profile.points[-1].distance_from_origin
    if total_distance <= 0:
        return "GREEN"
    
    total_weighted_score = 0
    total_weight = 0
    
    for seg in profile.segments:
        seg_length = seg.length(profile.points)
        seg_weight = seg_length / total_distance  # Fraction of total route
        
        # Calculate segment score based on gradient and features
        seg_score = seg._calculate_segment_score(profile.points, spot_system)
        
        # Weight by segment length
        total_weighted_score += seg_score * seg_weight
        total_weight += seg_weight
    
    # Final score is the weighted average
    final_score = total_weighted_score / total_weight if total_weight > 0 else 0
    
    print(f"Final score is {final_score:.2f}")
    
    return RouteDifficulty.from_score(final_score, spot_system).name