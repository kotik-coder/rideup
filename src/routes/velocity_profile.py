"""
velocity_profile.py
Calculates theoretical velocity profiles for mountain bike routes based on
physics models, rider parameters, and route characteristics.
"""

from dataclasses import dataclass
from typing import Any, List, Dict, Optional
import numpy as np
import math

from src.routes.profile_analyzer import Profile, ProfilePoint
from src.routes.terrain_and_weather import TerrainAnalysis, WeatherData

# ============ PHYSICS CONSTANTS ============
GRAVITY = 9.81  # m/s²
AIR_DENSITY = 1.2  # kg/m³
MIN_VELOCITY_DT = 1.0  # m/s (minimum for time step calculation)
WHEEL_CIRCUMFERENCE = 2.1  # meters (29" wheel)

# ============ RIDER PHYSIOLOGY CONSTANTS ============
MAX_FATIGUE = 100.0  # %
RECOVERY_RATE = 0.5  # % per second when coasting  
FATIGUE_RATE = 2.0   # % per second when at max power
OPTIMAL_CADENCE = 85.0  # RPM
CADENCE_EFFICIENCY_WIDTH = 25.0  # ±RPM around optimal for 100% efficiency
MIN_CADENCE_EFFICIENCY = 0.3  # Minimum efficiency at extreme cadences

# ============ GEAR SYSTEM CONSTANTS ============
MIN_GEAR_RATIO = 0.5  # e.g., 32T chainring / 52T cog
MAX_GEAR_RATIO = 2.5  # e.g., 32T / 12T

# ============ SUSPENSION CONSTANTS ============
BASE_SUSPENSION_LOSS = 0.03  # 3% on smooth terrain
MAX_SUSPENSION_LOSS = 0.15   # 15% on very rough terrain
MAX_SUSPENSION_LOSS_CAP = 0.95  # Cap at 95% loss

# ============ TRACTION & TURNING CONSTANTS ============
DEFAULT_TRACTION_MU = 0.8  # Coefficient of friction for good trail conditions
MAX_TURN_DECELERATION = 4.0  # m/s² for sharp turns
MIN_TURN_ANGLE = 5.0  # degrees (no deceleration below this)
TURN_SPEED_REFERENCE = 15.0  # m/s for speed factor normalization

# ============ ACCELERATION LIMITS ============
MAX_ACCELERATION = 3.0  # m/s² (realistic for MTB)
MAX_BRAKING = -5.0  # m/s² (hard braking)

# ============ SPEED LIMITS BY GRADIENT ============
STEEP_CLIMB_LIMIT = 5.0  # m/s (~18 km/h) for >10% gradient
MODERATE_CLIMB_LIMIT = 8.0  # m/s (~29 km/h) for >5% gradient  
STEEP_DESCENT_LIMIT = 12.0  # m/s (~43 km/h) for >8% descent
VERY_STEEP_DESCENT_LIMIT = 15.0  # m/s (~54 km/h) for >15% descent

# ============ POWER MANAGEMENT CONSTANTS ============
STEEP_DESCENT_POWER_FACTOR = 0.2  # 20% of sustained power on steep descents
MIN_DESCENT_POWER = 50.0  # watts (minimal pedaling)
HIGH_FATIGUE_POWER_FACTOR = 0.6  # 60% power when very fatigued
MODERATE_FATIGUE_POWER_FACTOR = 0.8  # 80% power when moderately fatigued
POWER_SPRINT_THRESHOLD = 1.2  # Use max power when needed power > 120% sustained

# ============ ROUGHNESS CALCULATION CONSTANTS ============
GRADIENT_CHANGE_SCALE = 20.0  # Scale factor for gradient-based roughness
SIGNIFICANT_DESCENT_GRADIENT = -0.05  # 5% descent for recovery

@dataclass
class RiderParameters:
    """Physical parameters for velocity modeling"""
    mass_rider: float = 85.0  # kg
    mass_bike: float = 15.0   # kg
    max_power: float = 800.0  # watts (short bursts)
    sustained_power: float = 250.0  # watts (realistic sustained)
    Crr: float = 0.02  # Typical MTB on trail
    CdA: float = 0.55   # Upright MTB position
    
    @property
    def total_mass(self) -> float:
        return self.mass_rider + self.mass_bike

@dataclass
class EnvironmentalParameters:
    """Environmental conditions for velocity modeling"""
    air_density: float = AIR_DENSITY
    gravity: float = GRAVITY
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
                env_params: EnvironmentalParameters,
                terrain_data: Optional[TerrainAnalysis] = None,
                weather_data: Optional[WeatherData] = None,
                gradient_blend_factor: float = 0.7):
        self.profile = profile
        self.rider = rider_params
        self.env = env_params        
        self.terrain_data = terrain_data
        self.weather_data = weather_data
        self.gradient_blend_factor = gradient_blend_factor  # 0.0 = all baseline, 1.0 = all real
        self.velocity_points: List[VelocityPoint] = []
        self._bearing_changes = self._calculate_bearing_changes()
        
        # Gear shifting state
        self.current_gear_ratio = (MIN_GEAR_RATIO + MAX_GEAR_RATIO) / 2
        
        # Previous gradient for suspension roughness estimation
        self.previous_gradient = 0.0
        
        # Fatigue state with enhanced tracking
        self.fatigue_level = 0.0
        self.power_history = []  # Track recent power output for fatigue calculation
        self.time_in_power_zone = 0.0  # Time spent in high power zones
        
    def _get_effective_gradient(self, point: ProfilePoint, previous_point: Optional[ProfilePoint] = None) -> float:
        """
        Calculate effective gradient by blending real and baseline gradients
        to balance accuracy with stability.
        """
        if previous_point is None:
            # First point, use baseline only for stability
            return point.baseline_gradient
        
        # Blend real gradient with baseline gradient
        real_gradient = point.gradient if hasattr(point, 'gradient') else point.baseline_gradient
        effective_gradient = (self.gradient_blend_factor * real_gradient + 
                             (1 - self.gradient_blend_factor) * point.baseline_gradient)
        
        return effective_gradient
    
    def _update_fatigue_model(self, power_output: float, power_required: float, 
                            dt: float, gradient: float) -> float:
        """
        Enhanced fatigue model that accounts for power bursts and recovery patterns.
        Returns updated fatigue level.
        """
        # Track recent power output for fatigue calculation
        self.power_history.append((power_output, dt))
        # Keep only last 60 seconds of history
        while sum(dt for _, dt in self.power_history) > 60:
            self.power_history.pop(0)
        
        # Calculate recent power intensity
        recent_power_avg = sum(p * t for p, t in self.power_history) / sum(t for _, t in self.power_history) if self.power_history else 0
        
        # Base fatigue rate depends on power output relative to sustained power
        power_ratio = power_output / self.rider.sustained_power
        
        if power_ratio > 1.5:
            # Sprint/peak power zone - very high fatigue rate
            fatigue_rate = FATIGUE_RATE * 3.0
            self.time_in_power_zone += dt
        elif power_ratio > 1.2:
            # High power zone - elevated fatigue rate
            fatigue_rate = FATIGUE_RATE * 2.0
            self.time_in_power_zone += dt
        elif power_ratio > 1.0:
            # Above sustained power - moderate fatigue rate
            fatigue_rate = FATIGUE_RATE * 1.5
            self.time_in_power_zone += dt * 0.5
        else:
            # Below sustained power - reduced fatigue rate
            fatigue_rate = FATIGUE_RATE * 0.8
            self.time_in_power_zone = max(0, self.time_in_power_zone - dt * 0.2)
        
        # Additional fatigue from recent high-intensity efforts
        if self.time_in_power_zone > 30:  # More than 30 seconds in high power zones
            fatigue_rate *= 1.5  # Increased fatigue due to accumulated effort
        elif self.time_in_power_zone > 60:
            fatigue_rate *= 2.0  # Severe fatigue after prolonged effort
        
        # Recovery on descents and easy sections
        if gradient < SIGNIFICANT_DESCENT_GRADIENT:
            # Enhanced recovery on descents
            recovery_multiplier = 1.0 + abs(gradient) * 5.0  # More recovery on steeper descents
            fatigue_level = max(0.0, self.fatigue_level - RECOVERY_RATE * recovery_multiplier * dt)
        elif power_output < self.rider.sustained_power * 0.7:
            # Moderate recovery when output is low
            fatigue_level = max(0.0, self.fatigue_level - RECOVERY_RATE * 0.5 * dt)
        else:
            # Fatigue accumulation
            fatigue_level = min(MAX_FATIGUE, self.fatigue_level + fatigue_rate * dt)
        
        return fatigue_level
        
    def _calculate_cadence(self, velocity: float, gear_ratio: float) -> float:
        """Calculate pedaling cadence in RPM"""
        if velocity <= 0.1:
            return 0.0
        # distance per revolution = gear_ratio * wheel_circumference
        revs_per_second = velocity / (gear_ratio * WHEEL_CIRCUMFERENCE)
        return revs_per_second * 60  # RPM

    def _get_cadence_efficiency(self, cadence: float) -> float:
        """Calculate efficiency penalty based on cadence (0.0 to 1.0)"""
        if cadence <= 0:
            return 0.0
        
        # Triangular efficiency curve
        if cadence < OPTIMAL_CADENCE - CADENCE_EFFICIENCY_WIDTH:
            # Below efficient range
            efficiency = (cadence - (OPTIMAL_CADENCE - 2*CADENCE_EFFICIENCY_WIDTH)) / CADENCE_EFFICIENCY_WIDTH
        elif cadence > OPTIMAL_CADENCE + CADENCE_EFFICIENCY_WIDTH:
            # Above efficient range
            efficiency = ((OPTIMAL_CADENCE + 2*CADENCE_EFFICIENCY_WIDTH) - cadence) / CADENCE_EFFICIENCY_WIDTH
        else:
            # Within efficient range
            efficiency = 1.0
        
        return max(MIN_CADENCE_EFFICIENCY, min(1.0, efficiency))

    def _auto_shift_gears(self, velocity: float, target_power: float, gradient: float) -> float:
        """Automatically select gear ratio to stay near optimal cadence"""
        if velocity <= 0.1:
            return MIN_GEAR_RATIO
        
        # Target cadence based on effort (higher power → slightly higher cadence)
        target_cadence = OPTIMAL_CADENCE
        if target_power > self.rider.sustained_power * 1.5:
            target_cadence += 10  # Sprinting = higher cadence
        
        # Calculate ideal gear ratio for target cadence
        ideal_ratio = velocity / (target_cadence / 60 * WHEEL_CIRCUMFERENCE)
        
        # Clamp to available range
        return max(MIN_GEAR_RATIO, min(MAX_GEAR_RATIO, ideal_ratio))
    
    def _calculate_bearing_changes(self) -> List[float]:
        """Calculate bearing changes between consecutive points"""
        bearing_changes = [0.0]  # First point has no bearing change
        
        points = self.profile.points
        for i in range(1, len(points)):
            if i < len(points) - 1:
                # Calculate bearing from previous to current point
                bearing_prev = points[i-1].bearing_to(points[i])
                # Calculate bearing from current to next point  
                bearing_next = points[i].bearing_to(points[i+1])
                
                # Calculate the absolute bearing change (handling 0-360 wrap-around)
                bearing_change = abs(bearing_next - bearing_prev)
                if bearing_change > 180:
                    bearing_change = 360 - bearing_change
                
                bearing_changes.append(bearing_change)
            else:
                bearing_changes.append(0.0)  # Last point
        
        return bearing_changes
    
    def _get_turning_deceleration(self, bearing_change: float, current_velocity: float) -> float:
        """
        Calculate additional deceleration required for turning based on bearing change
        and current velocity. More aggressive turns at higher speeds require more braking.
        """
        if bearing_change < MIN_TURN_ANGLE:
            return 0.0
        
        # Base deceleration based on bearing change (degrees to radians)
        turn_severity = math.radians(bearing_change) / math.pi
        
        # Velocity-dependent factor (higher speed = more deceleration needed)
        speed_factor = min(1.0, current_velocity / TURN_SPEED_REFERENCE)
        
        return turn_severity * speed_factor * MAX_TURN_DECELERATION
        
    def _calculate_suspension_loss_factor(self, velocity: float, gradient: float) -> float:
        """Calculate additional power loss due to suspension movement"""
        base_loss = BASE_SUSPENSION_LOSS
        
        # Roughness from gradient change (simulate bumps)
        gradient_change = abs(gradient - self.previous_gradient)
        roughness_factor = min(1.0, gradient_change * GRADIENT_CHANGE_SCALE)
        
        # Use terrain-based roughness if available
        if self.terrain_data and hasattr(self.terrain_data, 'get_roughness_score'):
            try:
                # Get roughness from terrain analysis
                terrain_roughness = self.terrain_data.get_roughness_score()
                # Combine gradient-based and terrain-based roughness
                total_roughness = max(roughness_factor, terrain_roughness)
            except (AttributeError, TypeError):
                total_roughness = roughness_factor
        else:
            total_roughness = roughness_factor
        
        # Interpolate loss between base and max
        suspension_loss = base_loss + (MAX_SUSPENSION_LOSS - base_loss) * total_roughness
        
        return min(MAX_SUSPENSION_LOSS_CAP, suspension_loss)
    
    def _get_available_power(self, fatigue_level: float, gradient: float, 
                           power_required: float, cadence_efficiency: float) -> float:
        """
        Calculate available power considering fatigue, gradient, and cadence efficiency.
        """
        # Base power availability based on fatigue
        if fatigue_level > 90.0:
            # Extreme fatigue - very limited power
            base_power = self.rider.sustained_power * 0.4
        elif fatigue_level > 75.0:
            # High fatigue - reduced power
            base_power = self.rider.sustained_power * HIGH_FATIGUE_POWER_FACTOR
        elif fatigue_level > 50.0:
            # Moderate fatigue
            base_power = self.rider.sustained_power * MODERATE_FATIGUE_POWER_FACTOR
        else:
            # Fresh - full sustained power
            base_power = self.rider.sustained_power
        
        # Adjust for gradient
        if gradient < -0.1:  # Steep descent
            base_power = min(base_power * STEEP_DESCENT_POWER_FACTOR, MIN_DESCENT_POWER)
        
        # Check if sprint power is needed and available
        if (power_required > base_power * POWER_SPRINT_THRESHOLD and 
            fatigue_level < 70.0 and  # Only sprint if not too fatigued
            self.time_in_power_zone < 45.0):  # Limit sprinting after prolonged effort
            
            # Sprint power available, but reduced by recent efforts
            sprint_power = self.rider.max_power * (1.0 - min(0.5, self.time_in_power_zone / 90.0))
            base_power = max(base_power, sprint_power)
        
        # Apply cadence efficiency
        return base_power * cadence_efficiency
            
    def calculate_velocity_profile(self) -> List[VelocityPoint]:
        """
        Calculate velocity profile using proper physics with realistic power constraints,
        turning dynamics, gear shifting, and suspension losses.
        """
        current_velocity = 2.0  # Start at walking speed (2 m/s = 7.2 km/h)
        self.velocity_points = []
        points = self.profile.points
        
        # Reset fatigue state
        self.fatigue_level = 0.0
        self.power_history = []
        self.time_in_power_zone = 0.0
        
        # Add initial point
        if points:
            initial_gradient = self._get_effective_gradient(points[0])
            initial_point = VelocityPoint(
                distance=points[0].distance_from_origin,
                velocity=current_velocity,
                acceleration=0.0,
                power_required=self._calculate_power_required(current_velocity, initial_gradient),
                power_available=self.rider.sustained_power,
                gradient=initial_gradient
            )
            self.velocity_points.append(initial_point)
            self.previous_gradient = initial_gradient
        
        for i in range(1, len(points)):
            point = points[i]
            previous_point = points[i-1] if i > 0 else None
            
            # Use hybrid gradient approach
            effective_gradient = self._get_effective_gradient(point, previous_point)
            θ = np.arctan(effective_gradient)
            
            # Calculate distance step
            dx = ProfilePoint.distance_between(points, i-1, i)
            if dx <= 0:
                self.previous_gradient = effective_gradient
                continue
                
            # Calculate time step based on current velocity
            dt = dx / max(current_velocity, MIN_VELOCITY_DT)
            
            # Auto-shift gears based on velocity and power demand
            estimated_power_needed = self._calculate_power_required(current_velocity, effective_gradient)
            self.current_gear_ratio = self._auto_shift_gears(current_velocity, estimated_power_needed, effective_gradient)
            
            # Calculate cadence and efficiency
            cadence = self._calculate_cadence(current_velocity, self.current_gear_ratio)
            cadence_efficiency = self._get_cadence_efficiency(cadence)
            
            # Calculate power required
            power_required = self._calculate_power_required(current_velocity, effective_gradient)
            
            # Update fatigue with enhanced model
            self.fatigue_level = self._update_fatigue_model(
                power_required, power_required, dt, effective_gradient
            )
            
            # Determine available power considering fatigue and recent efforts
            power_available = self._get_available_power(
                self.fatigue_level, effective_gradient, power_required, cadence_efficiency
            )
            
            # Calculate resistance forces using effective gradient
            F_rolling = self.rider.total_mass * self.env.gravity * self.rider.Crr * np.cos(θ)
            F_gravity = self.rider.total_mass * self.env.gravity * np.sin(θ)
            F_drag = 0.5 * self.env.air_density * self.rider.CdA * (current_velocity + self.env.headwind)**2
            
            # Calculate maximum propulsive force (power and traction limited)
            max_propulsive_force_from_power = power_available / max(current_velocity, 0.1)
            
            # Traction limit - use terrain data if available
            traction_mu = DEFAULT_TRACTION_MU
            if self.terrain_data and self.weather_data:
                try:
                    # Use the terrain's method to get adjusted traction
                    traction_score = self.terrain_data.get_adjusted_traction(self.weather_data)
                    # Convert traction score (0-1) to friction coefficient (0.3-0.9)
                    traction_mu = 0.3 + traction_score * 0.6
                except (AttributeError, TypeError) as e:
                    # Fallback to basic weather adjustment
                    if self.weather_data and self.weather_data.condition.lower() in ['rain', 'snow']:
                        traction_mu = DEFAULT_TRACTION_MU * 0.7
            elif self.weather_data and self.weather_data.condition.lower() in ['rain', 'snow']:
                # Basic weather adjustment if no terrain data
                traction_mu = DEFAULT_TRACTION_MU * 0.7

            max_traction_force = traction_mu * self.rider.total_mass * self.env.gravity * np.cos(θ)
            
            propulsive_force = min(max_propulsive_force_from_power, max_traction_force)
            
            # Calculate net force
            F_net = propulsive_force - F_rolling - F_gravity - F_drag
            
            # Apply turning deceleration
            turning_decel = self._get_turning_deceleration(
                self._bearing_changes[i] if i < len(self._bearing_changes) else 0.0,
                current_velocity
            )
            F_turning = self.rider.total_mass * turning_decel
            F_net -= F_turning
            
            # Calculate acceleration
            acceleration = F_net / self.rider.total_mass
            
            # Apply realistic acceleration limits
            acceleration = np.clip(acceleration, MAX_BRAKING, MAX_ACCELERATION)
            
            # Update velocity
            new_velocity = max(0.0, current_velocity + acceleration * dt)
            
            # Ensure realistic speed limits based on effective gradient
            if effective_gradient > 0.1:
                new_velocity = min(new_velocity, STEEP_CLIMB_LIMIT)
            elif effective_gradient > 0.05:
                new_velocity = min(new_velocity, MODERATE_CLIMB_LIMIT)
            elif effective_gradient < -0.15:
                new_velocity = min(new_velocity, VERY_STEEP_DESCENT_LIMIT)
            elif effective_gradient < -0.08:
                new_velocity = min(new_velocity, STEEP_DESCENT_LIMIT)
            
            # Apply suspension losses to power_required
            suspension_loss_factor = self._calculate_suspension_loss_factor(current_velocity, effective_gradient)
            power_required_with_loss = power_required * (1.0 + suspension_loss_factor)
            
            # Create velocity point
            vel_point = VelocityPoint(
                distance=point.distance_from_origin,
                velocity=new_velocity,
                acceleration=acceleration,
                power_required=power_required_with_loss,
                power_available=power_available,
                gradient=effective_gradient  # Store the effective gradient used
            )
            
            self.velocity_points.append(vel_point)
            current_velocity = new_velocity
            self.previous_gradient = effective_gradient
        
        return self.velocity_points

    def _calculate_power_required(self, velocity: float, gradient: float, crr: float = None) -> float:
        """
        Calculate power required to maintain given velocity at gradient
        """
        if crr is None:
            crr = self.rider.Crr
            
        θ = np.arctan(gradient)
        F_rolling = self.rider.total_mass * self.env.gravity * crr * np.cos(θ)
        F_gravity = self.rider.total_mass * self.env.gravity * np.sin(θ)
        F_drag = 0.5 * self.env.air_density * self.rider.CdA * (velocity + self.env.headwind)**2
        
        return (F_rolling + F_gravity + F_drag) * velocity
    
    def get_segment_statistics(self) -> Dict[str, Any]:
        """Calculate statistics for each segment of the route"""
        if not self.velocity_points:
            return {}
        
        # Calculate time for each segment
        total_time = 0.0
        for i in range(1, len(self.velocity_points)):
            dx = self.velocity_points[i].distance - self.velocity_points[i-1].distance
            avg_velocity = (self.velocity_points[i-1].velocity + self.velocity_points[i].velocity) / 2
            if avg_velocity > 0:
                total_time += dx / avg_velocity
        
        segment_stats = {
            'total_distance': self.velocity_points[-1].distance if self.velocity_points else 0,
            'total_time': total_time,
            'avg_velocity': np.mean([vp.velocity for vp in self.velocity_points]) if self.velocity_points else 0,
            'max_velocity': max([vp.velocity for vp in self.velocity_points]) if self.velocity_points else 0,
            'avg_power': np.mean([vp.power_required for vp in self.velocity_points]) if self.velocity_points else 0,
            'max_power': max([vp.power_required for vp in self.velocity_points]) if self.velocity_points else 0,
            'segments': []
        }
        
        return segment_stats