from typing import Any, List, Dict
import numpy as np

from src.routes.spot import Spot
from src.routes.profile_analyzer import SegmentProfile, StaticProfile
from src.routes.route_processor import ProcessedRoute
from src.routes.track import Track
from src.routes.trail_features import *
from src.routes.terrain_and_weather import TerrainAnalysis, WeatherData

def generate_route_profiles(spot: Spot, proute: ProcessedRoute, associated_tracks: List[Track]):
    static_profile = StaticProfile(spot.system, proute)
    segment_profile = SegmentProfile(static_profile, proute)
    
    # New statistics
    stats = RouteStatistics(segment_profile)
    weather_mod = _calculate_weather_modifier(spot.weather)
    surface_score = _calculate_surface_score(segment_profile.segments[0], spot.terrain)
    
    difficulty = _determine_difficulty(
        segment_profile, 
        static_profile,
        spot.system
    )
    
    return {
        'static': static_profile,
        'segments': segment_profile,
        'difficulty': difficulty,
        'statistics': stats,
        'weather_modifier': weather_mod,
        'surface_score': surface_score,
        'plot_data': [s.get_plot_data() for s in segment_profile.segments]
    }

def _calculate_weather_modifier(weather: WeatherData) -> float:
    """Return multiplier based on weather conditions"""
    if not weather:
        return 1.0
        
    precip_class = weather.precipitation_classification
    if precip_class['current_rate_violent'] or precip_class['forecast_violent']:
        return 1.5
    elif precip_class['current_rate_heavy']:
        return 1.3
    elif precip_class['current_rate_moderate']:
        return 1.15
    return 1.0

def _calculate_surface_score(segment: ElevationSegment, terrain: TerrainAnalysis) -> float:
    """Calculate difficulty contribution from surface types"""
    if not terrain.surface_types:
        return 0
        
    # Get surface weights from terrain analysis
    surface_score = sum(
        proportion * TerrainAnalysis.SURFACE_WEIGHTS.get(surface.lower(), 1.0)
        for surface, proportion in terrain.surface_types.items()
    )
    return (1 - surface_score) * 5  # Invert so lower traction = higher difficulty

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
        
        # Enhanced feature scoring using RatingSystem parameters
        if seg.feature_type:
            feature_config = seg.feature_type.get_config(spot_system)
            feature_score += feature_config.get('difficulty_impact', 0) * (seg.length()/100)
            
        # Score short features
        for sf in seg.short_features:
            sf_config = sf.feature_type.get_config(spot_system)
            feature_score += sf_config.get('difficulty_impact', 0) * (sf.length/10)
    
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

class RouteStatistics:
    def __init__(self, segment_profile: SegmentProfile):
        self.total_distance = segment_profile.segments[-1].distances[-1]
        self.elevation_gain = self._calculate_elevation_gain(segment_profile)
        self.feature_counts = self._count_features(segment_profile)
        
    def _calculate_elevation_gain(self, segment_profile):
        return sum(
            max(0, seg.gradients[i] * (seg.distances[i] - seg.distances[i-1]))
            for seg in segment_profile.segments
            for i in range(1, len(seg.gradients))
        )
        
    def _count_features(self, segment_profile):
        counts = {ft.name: 0 for ft in TrailFeatureType}
        for seg in segment_profile.segments:
            if seg.feature_type:
                counts[seg.feature_type.name] += 1
            for sf in seg.short_features:
                counts[sf.feature_type.name] += 1
        return counts