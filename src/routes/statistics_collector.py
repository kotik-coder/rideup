from typing import Any, List, Dict, Optional
import numpy as np

from src.routes.spot import Spot
from src.routes.profile_analyzer import Profile, Segment, StaticProfilePoint
from src.routes.route_processor import ProcessedRoute
from src.routes.track import Track
from src.routes.trail_features import *
from src.routes.terrain_and_weather import TerrainAnalysis, WeatherData

def generate_route_profiles(spot: Spot, proute: ProcessedRoute, associated_tracks: List[Track]) -> Dict[str, Any]:
    """Generate complete route analysis including profile and statistics"""
    profile = Profile(spot.system, proute)
    
    # Calculate additional statistics
    weather_mod = _calculate_weather_modifier(spot.weather)
    surface_score = _calculate_surface_score(profile.segments[0], spot.terrain) if profile.segments else 0
    difficulty = _determine_difficulty(profile, spot.system)
    
    stats = RouteStatistics(profile)
    
    return {
        'profile': profile,
        'difficulty': difficulty,
        'statistics': stats,
        'weather_modifier': weather_mod,
        'surface_score': surface_score
    }

def _calculate_weather_modifier(weather: Optional[WeatherData]) -> float:
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

def _calculate_surface_score(segment: Segment, terrain: TerrainAnalysis) -> float:
    """Calculate difficulty contribution from surface types"""
    if not terrain.surface_types:
        return 0
        
    # Get surface weights from terrain analysis
    surface_score = sum(
        proportion * TerrainAnalysis.SURFACE_WEIGHTS.get(surface.lower(), 1.0)
        for surface, proportion in terrain.surface_types.items()
    )
    return (1 - surface_score) * 5  # Invert so lower traction = higher difficulty

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
    total_distance = profile.segments[-1].distances[-1]
    elev_gain = max(p.elevation for seg in profile.segments for p in seg.points) - \
               min(p.elevation for seg in profile.segments for p in seg.points)
    
    # Normalize elevation gain (100m per km is considered challenging)
    elevation_score = min((elev_gain / total_distance) * 1000, 10)  # Max 10 points
    
    # Analyze segments
    for seg in profile.segments:
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
    """Collects and calculates various statistics about the route"""
    def __init__(self, profile: Profile):
        self.total_distance = profile.segments[-1].distances[-1] if profile.segments else 0
        self.elevation_gain = self._calculate_elevation_gain(profile)
        self.feature_counts = self._count_features(profile)
        self.segment_stats = self._calculate_segment_stats(profile)
        
    def _calculate_elevation_gain(self, profile: Profile) -> float:
        """Calculate total positive elevation gain"""
        if not profile.segments:
            return 0
            
        total_gain = 0.0
        prev_elevation = profile.segments[0].points[0].elevation
        
        for seg in profile.segments:
            for point in seg.points:
                if point.elevation > prev_elevation:
                    total_gain += point.elevation - prev_elevation
                prev_elevation = point.elevation
                
        return total_gain
        
    def _count_features(self, profile: Profile) -> Dict[str, int]:
        """Count occurrences of each feature type"""
        counts = {ft.name: 0 for ft in TrailFeatureType}
        
        for seg in profile.segments:
            if seg.feature_type:
                counts[seg.feature_type.name] += 1
            for sf in seg.short_features:
                counts[sf.feature_type.name] += 1
                
        return counts
    
    def _calculate_segment_stats(self, profile: Profile) -> Dict[str, Any]:
        """Calculate statistics about different segment types"""
        stats = {
            'total_segments': len(profile.segments),
            'steep_ascents': 0,
            'steep_descents': 0,
            'technical_sections': 0,
            'flow_sections': 0
        }
        
        for seg in profile.segments:
            if seg.gradient_type == GradientSegmentType.STEEP_ASCENT:
                stats['steep_ascents'] += 1
            elif seg.gradient_type == GradientSegmentType.STEEP_DESCENT:
                stats['steep_descents'] += 1
                
            if seg.feature_type in [TrailFeatureType.TECHNICAL_ASCENT, TrailFeatureType.TECHNICAL_DESCENT]:
                stats['technical_sections'] += 1
            elif seg.feature_type in [TrailFeatureType.FLOW_DESCENT, TrailFeatureType.ROLLER]:
                stats['flow_sections'] += 1
                
        return stats

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary format"""
        return {
            'total_distance': self.total_distance,
            'elevation_gain': self.elevation_gain,
            'feature_counts': self.feature_counts,
            'segment_stats': self.segment_stats
        }