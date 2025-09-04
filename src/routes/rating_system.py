from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class RatingSystem:
    """Central configuration for all trail parameters including features and gradients"""
    # Gradient thresholds
    gradient_thresholds: Dict['GradientSegmentType', Tuple[float, float]] = field(
        default_factory=lambda: {
            GradientSegmentType.ASCENT: (0.01, 0.10),
            GradientSegmentType.DESCENT: (-0.10, -0.01),
            GradientSegmentType.STEEP_ASCENT: (0.10, float('inf')),
            GradientSegmentType.STEEP_DESCENT: (-float('inf'), -0.10),
            GradientSegmentType.FLAT: (-0.01, 0.01)
        }
    )
    
    # Difficulty thresholds
    difficulty_thresholds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "GREEN": (0, 2.0), 
            "BLUE": (2.0, 7.0),
            "BLACK": (7.0, 15.0),
            "DOUBLE_BLACK": (15.0, float('inf'))
        }
    )

    # Feature parameters
    feature_parameters: Dict['TrailFeatureType', Dict[str, Any]] = field(
        default_factory=lambda: {
            TrailFeatureType.ROLLER: {
                "min_length": 50,
                "max_length": 250,
                "gradient_range": (-0.05, 0.05),
                "wavelength_range": (10, 50),  # Added for roller
                "difficulty_impact": 1.5
            },
            TrailFeatureType.FLOW_DESCENT: {
                "min_length": 50,
                "max_length": 500,
                "gradient_range": (-0.12, -0.025),
                "wavelength_range": (10, 50),  # Already existed
                "difficulty_impact": 1.2
            },
            TrailFeatureType.SWITCHBACK: {
                "min_length": 5,
                "max_length": 30,
                "gradient_range": (-0.25, -0.15),
                "angular_displacement": 120,
                "difficulty_impact": 2.0
            },
            TrailFeatureType.TECHNICAL_DESCENT: {
                "min_length": 10,
                "max_length": 250,
                "gradient_range": (-float('inf'), -0.15),
                "required_short_features": 2,
                "difficulty_impact": 3.0
            },
            TrailFeatureType.TECHNICAL_ASCENT: {
                "min_length": 10,
                "max_length": 250,
                "gradient_range": (0.15, float('inf')),
                "required_short_features": 2,
                "difficulty_impact": 2.5
            },
            TrailFeatureType.KICKER: {
                "min_length": 1,
                "max_length": 10,
                "gradient_range": (-0.3, -0.15),
                "difficulty_impact": 2.5
            },
            TrailFeatureType.DROP: {
                "min_length": 1,
                "max_length": 8,
                "gradient_range": (-float('inf'), -0.25),
                "difficulty_impact": 3.5
            }
        }
    )

    # Feature compatibility with gradient types
    feature_compatibility: Dict['GradientSegmentType', List['TrailFeatureType']] = field(
        default_factory=lambda: {
            GradientSegmentType.ASCENT: [TrailFeatureType.TECHNICAL_ASCENT],
            GradientSegmentType.DESCENT: [TrailFeatureType.TECHNICAL_DESCENT, TrailFeatureType.FLOW_DESCENT, TrailFeatureType.SWITCHBACK],
            GradientSegmentType.STEEP_ASCENT: [TrailFeatureType.TECHNICAL_ASCENT],
            GradientSegmentType.STEEP_DESCENT: [TrailFeatureType.TECHNICAL_DESCENT, TrailFeatureType.DROP, TrailFeatureType.KICKER],
            GradientSegmentType.FLAT: []
        }
    )

    # Segment identification parameters
    min_segment_length: float = 50  # Minimum length in meters
    min_segment_points: int = 5    # Minimum number of points
    min_steep_length: float = 10
    step_feature_max_length: float = 15
    
    # Wavelength parameters
    wavelength_clustering_eps: float = 0.5
    wavelength_match_tolerance: float = 0.3
    
    num_oscillations_threshold: int = 3
    
    feature_threshold: float = 0.7  # Minimum score for roller/flow features
    technical_threshold: float = 5.0  # Minimum score for technical features
    feature_clustering_eps: float = 5.0  # Meters between features to consider them clustered

    def validate_config(self) -> List[str]:
        """Check configuration consistency"""
        errors = []
        
        # Check gradient thresholds
        prev_max = None
        for name, (min_val, max_val) in sorted(self.gradient_thresholds.items()):
            if prev_max is not None and min_val < prev_max:
                errors.append(f"Gradient threshold overlap: {name}")
            prev_max = max_val
            
        # Check feature parameters
        for feature, params in self.feature_parameters.items():
            if params['min_length'] > params['max_length']:
                errors.append(f"Invalid length range for {feature}")
                
        return errors

    def get_feature_config(self, feature_type: 'TrailFeatureType') -> Dict[str, Any]:
        """Get configuration for a specific feature type"""
        return self.feature_parameters.get(feature_type, {})

    def get_compatible_features(self, gradient_type: 'GradientSegmentType') -> List['TrailFeatureType']:
        """Get features that can occur on this gradient type"""
        return self.feature_compatibility.get(gradient_type, [])

    def is_feature_compatible(self, feature_type: 'TrailFeatureType', gradient_type: 'GradientSegmentType') -> bool:
        """Check if a feature type is compatible with a gradient type"""
        return feature_type in self.feature_compatibility.get(gradient_type, [])

    @classmethod
    def create(cls, custom_config: Optional[Dict] = None) -> 'RatingSystem':
        """Factory method with enhanced feature support"""
        system = cls()
        
        if custom_config:
            # Handle nested feature configurations
            for key, value in custom_config.items():
                if key == "feature_parameters" and isinstance(value, dict):
                    system.feature_parameters.update(value)
                elif key == "feature_compatibility" and isinstance(value, dict):
                    system.feature_compatibility.update(value)
                elif hasattr(system, key):
                    setattr(system, key, value)
        
        return system

class GradientSegmentType(Enum):
    """Classification of gradient segments with enhanced transition logic"""
    ASCENT = auto()
    DESCENT = auto()
    STEEP_ASCENT = auto()
    STEEP_DESCENT = auto()
    FLAT = auto()
    
    def title(self):
        return self.name.replace('_', ' ').title()

    def get_thresholds(self, spot_system: RatingSystem) -> Tuple[float, float]:
        """Get thresholds from RatingSystem configuration"""
        return spot_system.gradient_thresholds[self]

    @classmethod
    def from_gradient(cls, gradient: float, spot_system: RatingSystem) -> 'GradientSegmentType':
        """Classify gradient using dynamic thresholds"""
        for segment_type in cls:
            min_grad, max_grad = segment_type.get_thresholds(spot_system)
            if min_grad <= gradient < max_grad:
                return segment_type
        return cls.FLAT

    def is_transitional_to(self, other: 'GradientSegmentType') -> bool:
        """Enhanced transition logic with priority rules"""
        transition_rules = {
            self.FLAT: [self.ASCENT, self.DESCENT],
            self.ASCENT: [self.STEEP_ASCENT, self.FLAT],
            self.DESCENT: [self.STEEP_DESCENT, self.FLAT],
            self.STEEP_ASCENT: [self.ASCENT],
            self.STEEP_DESCENT: [self.DESCENT]
        }
        return other in transition_rules.get(self, [])

class TrailFeatureType(Enum):
    """Enhanced trail feature classification with physical properties"""
    ROLLER = auto()
    SWITCHBACK = auto()
    TECHNICAL_DESCENT = auto()
    TECHNICAL_ASCENT = auto()
    FLOW_DESCENT = auto()
    KICKER = auto()
    DROP = auto()
    
    def title(self):
        return self.name.replace('_', ' ').title()

    def get_config(self, rating_system: RatingSystem) -> Dict[str, Any]:
        """Get feature parameters with validation"""
        config = rating_system.get_feature_config(self)
        # Ensure required fields exist
        config.setdefault('min_length', 1)
        config.setdefault('max_length', 100)
        config.setdefault('gradient_range', (-float('inf'), float('inf')))
        return config

    def is_compatible_with(self, gradient_type: GradientSegmentType, rating_system: RatingSystem) -> bool:
        """Check compatibility with additional physical constraints"""
        if not rating_system.is_feature_compatible(self, gradient_type):
            return False
            
        # Additional physical constraints
        if self == self.SWITCHBACK and gradient_type != GradientSegmentType.STEEP_DESCENT:
            return False
        if self == self.KICKER and gradient_type != GradientSegmentType.STEEP_ASCENT:
            return False
            
        return True

    def validate_segment_length(self, length: float, rating_system: RatingSystem) -> bool:
        """Validate if length is appropriate for this feature"""
        config = self.get_config(rating_system)
        return config['min_length'] <= length <= config['max_length']        

class RouteDifficulty(Enum):
    """Enhanced difficulty classification with physics-based thresholds"""
    GREEN = "GREEN"
    BLUE = "BLUE"
    BLACK = "BLACK"
    DOUBLE_BLACK = "DOUBLE_BLACK"

    def get_thresholds(self, spot_system: RatingSystem) -> Tuple[float, float]:
        """Get dynamic thresholds based on rating system"""
        return spot_system.difficulty_thresholds[self.value]

    @classmethod
    def from_score(cls, score: float, spot_system: RatingSystem) -> 'RouteDifficulty':
        """Classify with safety margins"""
        for difficulty in cls:
            min_thresh, max_thresh = difficulty.get_thresholds(spot_system)
            if min_thresh <= score < max_thresh * 0.95:  # 5% safety margin
                return difficulty
        return cls.DOUBLE_BLACK