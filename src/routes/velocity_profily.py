"""
velocity_profile.py
Calculates theoretical velocity profiles for mountain bike routes based on
physics models, rider parameters, and route characteristics.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from enum import Enum

from src.routes.rating_system import GradientSegmentType as gst
from src.routes.rating_system import TrailFeatureType as tft
from src.routes.profile_analyzer import Profile, ProfileSegment, ProfilePoint

class RiderAbility(Enum):
    BEGINNER = 0
    INTERMEDIATE = 1
    ADVANCED = 2
    EXPERT = 3
    PRO = 4

@dataclass
class RiderParameters:
    """Physical parameters for velocity modeling"""
    mass_rider: float = 70.0  # kg
    mass_bike: float = 15.0   # kg
    max_power: float = 400.0  # watts
    sustained_power: float = 250.0  # watts
    Crr: float = 0.008  # coefficient of rolling resistance
    CdA: float = 0.4    # drag coefficient * area (m²)
    ability: RiderAbility = RiderAbility.INTERMEDIATE
    
    @property
    def total_mass(self) -> float:
        return self.mass_rider + self.mass_bike

@dataclass
class EnvironmentalParameters:
    """Environmental conditions for velocity modeling"""
    air_density: float = 1.2  # kg/m³
    gravity: float = 9.81     # m/s²
    headwind: float = 0.0     # m/s

@dataclass
class VelocityPoint:
    """Velocity at a specific point along the route"""
    distance: float
    velocity: float
    acceleration: float
    power_required: float
    power_available: float
    gradient: float

class VelocityProfileCalculator:
    """
    Calculates theoretical velocity profiles for mountain bike routes
    using physics-based models and rider parameters.
    """
    
    def __init__(self, profile: Profile, rider_params: RiderParameters, 
                 env_params: EnvironmentalParameters):
        self.profile = profile
        self.rider = rider_params
        self.env = env_params
        self.velocity_points: List[VelocityPoint] = []
        
        # Ability-based speed modifiers
        self.ability_modifiers = {
            RiderAbility.BEGINNER: 0.6,
            RiderAbility.INTERMEDIATE: 0.8,
            RiderAbility.ADVANCED: 1.0,
            RiderAbility.EXPERT: 1.2,
            RiderAbility.PRO: 1.4
        }
        
        # Feature-based speed reduction factors
        self.feature_speed_factors = {
            tft.TECHNICAL_ASCENT: 0.5,
            tft.TECHNICAL_DESCENT: 0.6,
            tft.ROLLERCOASTER: 0.9,
            tft.FLOW_DESCENT: 1.1,
            tft.SWITCHBACK: 0.4,
            tft.KICKER: 0.7,
            tft.DROP: 0.3
        }
    
    def calculate_velocity_profile(self) -> List[VelocityPoint]:
        """
        Calculate the complete velocity profile for the route
        """
        print("Calculating velocity profile...")
        
        # Initialize with starting conditions
        current_velocity = 0.0
        self.velocity_points = []
        
        # Process each segment
        for segment in self.profile.segments:
            segment_velocities = self._process_segment(segment, current_velocity)
            
            if segment_velocities:
                current_velocity = segment_velocities[-1].velocity
                self.velocity_points.extend(segment_velocities)
        
        return self.velocity_points
    
    def _process_segment(self, segment: ProfileSegment, initial_velocity: float) -> List[VelocityPoint]:
        """
        Process an individual segment to calculate velocities
        """
        points = segment.get_points(self.profile.points)
        segment_velocities = []
        current_velocity = initial_velocity
        
        # Get segment-specific parameters
        speed_factor = self._get_segment_speed_factor(segment)
        max_power = self._get_segment_max_power(segment)
        
        for i in range(len(points) - 1):
            point = points[i]
            next_point = points[i + 1]
            
            # Calculate distance step
            dx = next_point.distance_from_origin - point.distance_from_origin
            
            # Calculate optimal velocity for this step
            optimal_velocity = self._calculate_optimal_velocity(
                point.gradient, current_velocity, dx, max_power
            )
            
            # Apply ability and feature modifiers
            final_velocity = optimal_velocity * speed_factor
            
            # Calculate acceleration
            acceleration = (final_velocity - current_velocity) / dx if dx > 0 else 0
            
            # Calculate power requirements
            power_req = self._calculate_power_required(final_velocity, point.gradient)
            power_avail = min(max_power, self.rider.max_power)
            
            # Create velocity point
            vel_point = VelocityPoint(
                distance=point.distance_from_origin,
                velocity=final_velocity,
                acceleration=acceleration,
                power_required=power_req,
                power_available=power_avail,
                gradient=point.gradient
            )
            
            segment_velocities.append(vel_point)
            current_velocity = final_velocity
        
        return segment_velocities
    
    def _calculate_optimal_velocity(self, gradient: float, current_velocity: float, 
                                  dx: float, max_power: float) -> float:
        """
        Calculate optimal velocity using physics equations
        """
        def power_equation(v):
            # Total power required = rolling resistance + gravity + acceleration + drag
            F_rolling = self.rider.total_mass * self.env.gravity * self.rider.Crr * np.cos(np.arctan(gradient))
            F_gravity = self.rider.total_mass * self.env.gravity * np.sin(np.arctan(gradient))
            F_drag = 0.5 * self.env.air_density * self.rider.CdA * (v + self.env.headwind)**2
            
            # For small dx, approximate acceleration power
            if dx > 0:
                accel_power = self.rider.total_mass * (v**2 - current_velocity**2) / (2 * dx)
            else:
                accel_power = 0
            
            total_power = (F_rolling + F_gravity + F_drag) * v + accel_power
            return total_power
        
        # Find velocity that doesn't exceed max power
        def power_constraint(v):
            return max_power - power_equation(v)
        
        # Try to find maximum sustainable velocity
        try:
            result = minimize_scalar(
                lambda v: -v,  # Maximize velocity
                bounds=(0, 30),  # Reasonable speed limits for MTB
                constraints={'type': 'ineq', 'fun': power_constraint},
                method='trust-constr'
            )
            return max(0, result.x)
        except:
            # Fallback: use analytical solution for flat terrain
            if abs(gradient) < 0.01:  # Nearly flat
                # Solve cubic equation for drag-limited speed
                a = 0.5 * self.env.air_density * self.rider.CdA
                b = 0
                c = self.rider.total_mass * self.env.gravity * self.rider.Crr
                d = -max_power
                
                # Find real positive root
                roots = np.roots([a, b, c, d])
                real_roots = [root.real for root in roots if abs(root.imag) < 1e-6 and root.real > 0]
                return min(real_roots) if real_roots else 5.0
            else:
                # Conservative estimate for steep terrain
                return 3.0 if gradient > 0 else 8.0
    
    def _calculate_power_required(self, velocity: float, gradient: float) -> float:
        """
        Calculate power required to maintain given velocity at gradient
        """
        F_rolling = self.rider.total_mass * self.env.gravity * self.rider.Crr * np.cos(np.arctan(gradient))
        F_gravity = self.rider.total_mass * self.env.gravity * np.sin(np.arctan(gradient))
        F_drag = 0.5 * self.env.air_density * self.rider.CdA * (velocity + self.env.headwind)**2
        
        return (F_rolling + F_gravity + F_drag) * velocity
    
    def _get_segment_speed_factor(self, segment: ProfileSegment) -> float:
        """
        Get speed reduction factor based on segment type and features
        """
        base_factor = self.ability_modifiers[self.rider.ability]
        
        # Apply feature-based modifiers
        if segment.feature:
            feature_factor = self.feature_speed_factors.get(segment.feature.feature_type, 1.0)
            base_factor *= feature_factor
        
        # Apply short features modifiers
        for short_feature in segment.short_features:
            short_factor = self.feature_speed_factors.get(short_feature.feature_type, 1.0)
            base_factor *= short_factor
        
        return base_factor
    
    def _get_segment_max_power(self, segment: ProfileSegment) -> float:
        """
        Get maximum available power for segment type
        """
        if segment.gradient_type in [gst.STEEP_ASCENT, gst.ASCENT]:
            # Reduced power on climbs
            return self.rider.sustained_power * 0.8
        elif segment.gradient_type in [gst.STEEP_DESCENT, gst.DESCENT]:
            # Coasting or minimal power on descents
            return min(self.rider.max_power * 0.3, 100)
        else:
            # Flat terrain - full power available
            return self.rider.max_power
    
    def get_segment_statistics(self) -> Dict:
        """
        Get statistics for each segment type
        """
        stats = {}
        
        for seg_type in gst:
            seg_velocities = [
                vp for vp in self.velocity_points 
                if self._get_point_segment_type(vp.distance) == seg_type
            ]
            
            if seg_velocities:
                velocities = [vp.velocity for vp in seg_velocities]
                stats[seg_type.name] = {
                    'avg_velocity': np.mean(velocities),
                    'max_velocity': np.max(velocities),
                    'min_velocity': np.min(velocities),
                    'distance': sum(vp.distance for vp in seg_velocities),
                    'time': sum(1/vp.velocity for vp in seg_velocities if vp.velocity > 0)
                }
        
        return stats
    
    def _get_point_segment_type(self, distance: float) -> gst:
        """
        Find which segment a distance point belongs to
        """
        for segment in self.profile.segments:
            start_dist = self.profile.points[segment.start_abs_idx].distance_from_origin
            end_dist = self.profile.points[segment.end_abs_idx].distance_from_origin
            
            if start_dist <= distance <= end_dist:
                return segment.gradient_type
        
        return gst.FLAT  # Default

# Example usage function
def create_velocity_profile(profile: Profile, 
                          rider_level: RiderAbility = RiderAbility.INTERMEDIATE) -> VelocityProfileCalculator:
    """
    Convenience function to create a velocity profile with default parameters
    """
    rider_params = RiderParameters(ability=rider_level)
    env_params = EnvironmentalParameters()
    
    calculator = VelocityProfileCalculator(profile, rider_params, env_params)
    calculator.calculate_velocity_profile()
    
    return calculator