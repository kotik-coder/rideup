from typing import Any, List, Dict, Optional
import numpy as np

from src.routes.spot import Spot
from src.routes.profile_analyzer import Profile, ProfileSegment
from src.routes.route_processor import ProcessedRoute
from src.routes.track import Track, TrackAnalysis, TrackPoint
from src.routes.rating_system import *
from src.routes.terrain_and_weather import TerrainAnalysis, WeatherData
from src.routes.velocity_profily import VelocityPoint, VelocityProfileCalculator, RiderAbility, RiderParameters, EnvironmentalParameters
from datetime import datetime

def generate_route_profiles(spot: Spot, proute: ProcessedRoute, associated_tracks: List[Track]) -> Dict[str, Any]:
    """Generate complete route analysis including profile and statistics"""
    profile = Profile(spot.system, proute)
    
    # Calculate additional statistics
    difficulty = _determine_difficulty(profile, spot.system)
    
    # Generate actual velocity profiles from GPS tracks
    actual_velocity_profiles = _generate_velocity_profiles(proute, associated_tracks)
    
    # Generate theoretical velocity profiles
    theoretical_velocity_profiles = _generate_theoretical_velocity_profiles(profile)
    
    return {
        'profile': profile,
        'difficulty': difficulty,
        'dynamic': {
            'actual': actual_velocity_profiles,
            'theoretical': theoretical_velocity_profiles
        }
    }

def _generate_velocity_profiles(proute: ProcessedRoute, tracks: List[Track]) -> List[TrackAnalysis]:
    """Generate velocity profiles from GPS tracks that match the route"""
    velocity_profiles = []
    
    if not tracks or not proute.smooth_points:
        return velocity_profiles
    
    total_route_distance = proute.total_distance()
    if total_route_distance <= 0:
        return velocity_profiles
    
    for track in tracks:
        if not track.points or len(track.points) < 2:
            continue
            
        # Only process tracks that are reasonably similar in length to the route
        track_distance = sum(
            track.points[i-1].point.distance_to(track.points[i].point)
            for i in range(1, len(track.points)))
        
        if abs(track_distance - total_route_distance) / total_route_distance > 0.5:
            continue
        
        # Use the track's built-in analysis if available
        if hasattr(track, 'analysis') and track.analysis:
            # Convert TrackAnalysis to be compatible with graph_generation
            velocity_profiles.extend(track.analysis)
        else:
            # Fallback: create basic velocity analysis from track points
            velocity_profiles.extend(_create_basic_velocity_analysis(track))
    
    return velocity_profiles

def _generate_theoretical_velocity_profiles(profile: Profile) -> Dict[str, Any]:
    """Generate theoretical velocity profiles for different rider abilities"""
    theoretical_profiles = {}
    
    # Generate profiles for different rider abilities
    for ability in RiderAbility:
        # Create velocity profile calculator
        rider_params = RiderParameters(ability=ability)
        env_params = EnvironmentalParameters()
        calculator = VelocityProfileCalculator(profile, rider_params, env_params)
        
        # Calculate velocity profile
        velocity_points = calculator.calculate_velocity_profile()
        
        # Convert to format compatible with graph generation
        converted_points = _convert_velocity_points_to_track_analysis(velocity_points)
        
        # Get segment statistics
        segment_stats = calculator.get_segment_statistics()
        
        theoretical_profiles[ability.name] = {
            'analysis_points': converted_points,
            'segment_statistics': segment_stats,
            'rider_ability': ability.name,
            'total_time': sum(1/vp.velocity for vp in velocity_points if vp.velocity > 0),
            'avg_velocity': np.mean([vp.velocity for vp in velocity_points]) if velocity_points else 0,
            'max_velocity': max([vp.velocity for vp in velocity_points]) if velocity_points else 0
        }
    
    return theoretical_profiles

def _convert_velocity_points_to_track_analysis(velocity_points: List[VelocityPoint]) -> List[TrackAnalysis]:
    """Convert VelocityPoint objects to TrackAnalysis format for compatibility"""
    track_analysis_points = []
    
    for vp in velocity_points:
        track_analysis = TrackAnalysis(
            horizontal_speed=vp.velocity,
            vertical_speed=0.0,  # Not calculated in theoretical model
            horizontal_accel=vp.acceleration,
            vertical_accel=0.0,   # Not calculated in theoretical model
            distance_from_start=vp.distance
        )
        track_analysis_points.append(track_analysis)
    
    return track_analysis_points

def _create_basic_velocity_analysis(track: Track) -> List[TrackAnalysis]:
    """Create basic velocity analysis from track points when no analysis exists"""
    analysis_points = []
    
    if not track.points or len(track.points) < 2:
        return analysis_points
    
    # Calculate cumulative distance
    distances = [0.0]
    for i in range(1, len(track.points)):
        dist = track.points[i-1].point.distance_to(track.points[i].point)
        distances.append(distances[-1] + dist)
    
    # Calculate velocities and accelerations
    for i in range(len(track.points)):
        if i == 0:
            # First point - use forward difference
            if len(track.points) > 1:
                dt = (track.points[1].timestamp - track.points[0].timestamp).total_seconds()
                if dt > 0:
                    h_speed = distances[1] / dt
                    v_speed = (track.points[1].point.elevation - track.points[0].point.elevation) / dt
                else:
                    h_speed = 0
                    v_speed = 0
            else:
                h_speed = 0
                v_speed = 0
            h_accel = 0
            v_accel = 0
            
        elif i == len(track.points) - 1:
            # Last point - use backward difference
            dt = (track.points[-1].timestamp - track.points[-2].timestamp).total_seconds()
            if dt > 0:
                h_speed = (distances[-1] - distances[-2]) / dt
                v_speed = (track.points[-1].point.elevation - track.points[-2].point.elevation) / dt
            else:
                h_speed = 0
                v_speed = 0
            h_accel = 0
            v_accel = 0
            
        else:
            # Middle points - use central difference
            dt_prev = (track.points[i].timestamp - track.points[i-1].timestamp).total_seconds()
            dt_next = (track.points[i+1].timestamp - track.points[i].timestamp).total_seconds()
            
            if dt_prev > 0 and dt_next > 0:
                h_speed_prev = (distances[i] - distances[i-1]) / dt_prev
                h_speed_next = (distances[i+1] - distances[i]) / dt_next
                h_speed = (h_speed_prev + h_speed_next) / 2
                
                v_speed_prev = (track.points[i].point.elevation - track.points[i-1].point.elevation) / dt_prev
                v_speed_next = (track.points[i+1].point.elevation - track.points[i].point.elevation) / dt_next
                v_speed = (v_speed_prev + v_speed_next) / 2
                
                # Simple acceleration calculation
                h_accel = (h_speed_next - h_speed_prev) / ((dt_prev + dt_next) / 2)
                v_accel = (v_speed_next - v_speed_prev) / ((dt_prev + dt_next) / 2)
            else:
                h_speed = 0
                v_speed = 0
                h_accel = 0
                v_accel = 0
        
        # Apply physical limits
        h_speed = min(h_speed, 47.2)  # MAX_PLAUSIBLE_SPEED
        v_speed = max(min(v_speed, 10.0), -10.0)  # MAX_VERTICAL_SPEED
        h_accel = max(min(h_accel, 10.0), -10.0)  # MAX_PLAUSIBLE_ACCEL
        v_accel = max(min(v_accel, 10.0), -10.0)  # MAX_PLAUSIBLE_ACCEL
        
        analysis_points.append(TrackAnalysis(
            horizontal_speed=h_speed,
            vertical_speed=v_speed,
            horizontal_accel=h_accel,
            vertical_accel=v_accel,
            distance_from_start=distances[i]
        ))
    
    return analysis_points

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