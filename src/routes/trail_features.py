from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
from src.routes.spot import RatingSystem

class GradientSegmentType(Enum):
    """Classification of gradient segments with enhanced transition logic"""
    ASCENT = "ASCENT"
    DESCENT = "DESCENT" 
    STEEP_ASCENT = "STEEP_ASCENT"
    STEEP_DESCENT = "STEEP_DESCENT"
    FLAT = "FLAT"

    def get_thresholds(self, spot_system: RatingSystem) -> Tuple[float, float]:
        """Get thresholds from RatingSystem configuration"""
        return spot_system.gradient_thresholds[self.value]

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

    def should_merge_with(self, other: 'GradientSegmentType') -> bool:
        """Determine if segments with these gradient types should merge"""
        return (self == other or 
                self.is_transitional_to(other))
    
    def get_merge_priority(self) -> int:
        """Priority for merging segments (higher = more important to keep separate)"""
        return {
            self.FLAT: 0,
            self.ASCENT: 1,
            self.DESCENT: 1,
            self.STEEP_ASCENT: 2,
            self.STEEP_DESCENT: 2
        }[self]

    def get_difficulty_modifier(self) -> float:
        """Returns a multiplier for difficulty calculations"""
        return {
            self.FLAT: 1.0,
            self.ASCENT: 1.2,
            self.DESCENT: 1.1,
            self.STEEP_ASCENT: 1.5,
            self.STEEP_DESCENT: 1.4
        }[self]

class TrailFeatureType(Enum):
    """Enhanced trail feature classification with physical properties"""
    ROLLER = auto()
    SWITCHBACK = auto()
    TECHNICAL_DESCENT = auto()
    TECHNICAL_ASCENT = auto()
    FLOW_DESCENT = auto()
    KICKER = auto()
    DROP = auto()

    def get_config(self, rating_system: RatingSystem) -> Dict[str, Any]:
        """Get feature parameters with validation"""
        config = rating_system.get_feature_config(self.name)
        # Ensure required fields exist
        config.setdefault('min_length', 1)
        config.setdefault('max_length', 100)
        config.setdefault('gradient_range', (-float('inf'), float('inf')))
        return config

    def is_compatible_with(self, gradient_type: GradientSegmentType, rating_system: RatingSystem) -> bool:
        """Check compatibility with additional physical constraints"""
        if not rating_system.is_feature_compatible(self.name, gradient_type.name):
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
    
    def get_optimal_length(self, rating_system: RatingSystem) -> float:
        """Get recommended length for this feature"""
        config = self.get_config(rating_system)
        return min(config['max_length'], 
                 max(config['min_length'], config.get('optimal_length', config['min_length'])))

    def is_technical(self) -> bool:
        """Check if feature is considered technically challenging"""
        return self in [
            self.TECHNICAL_DESCENT, 
            self.TECHNICAL_ASCENT,
            self.DROP,
            self.KICKER
        ]

    def get_energy_cost(self, length: float, gradient: float) -> float:
        """Estimate energy cost for this feature type"""
        base_cost = {
            self.ROLLER: 1.2,
            self.FLOW_DESCENT: 1.0,
            self.TECHNICAL_DESCENT: 1.8,
            self.TECHNICAL_ASCENT: 2.0,
            self.SWITCHBACK: 2.2,
            self.KICKER: 1.5,
            self.DROP: 1.7
        }[self]
        
        return base_cost * length * (1 + abs(gradient))

@dataclass
class ShortFeature:
    """Enhanced localized feature with physics-based validation"""
    feature_type: TrailFeatureType
    gradient_type: GradientSegmentType
    start_index: int  # Relative to segment start
    end_index: int    # Relative to segment start 
    max_gradient: float
    length: float
    _validated: Optional[bool] = None

    def get_absolute_indices(self, segment_start: int) -> Tuple[int, int]:
        """Convert to absolute indices in the profile"""
        return (segment_start + self.start_index, 
                segment_start + self.end_index)

    def validate(self, rating_system: RatingSystem) -> bool:
        """Cached validation with physical constraints"""
        if self._validated is not None:
            return self._validated
            
        config = self.feature_type.get_config(rating_system)
        
        # Length validation
        valid_length = (config['min_length'] <= self.length <= config['max_length'])
        
        # Gradient validation
        min_grad, max_grad = config['gradient_range']
        valid_gradient = (min_grad <= self.max_gradient <= max_grad)
        
        # Compatibility check
        valid_combo = self.feature_type.is_compatible_with(self.gradient_type, rating_system)
        
        self._validated = valid_length and valid_gradient and valid_combo
        return self._validated

    def get_difficulty_score(self, rating_system: RatingSystem) -> float:
        """Calculate dynamic difficulty score"""
        if not self.validate(rating_system):
            return 0.0
            
        config = self.feature_type.get_config(rating_system)
        base_score = config.get('difficulty_impact', 1.0)
        
        # Gradient modifier
        grad_mod = 1 + abs(self.max_gradient - config['gradient_range'][0]) * 2
        
        # Length modifier
        length_mod = min(2.0, self.length / config['min_length'])
        
        return base_score * grad_mod * length_mod

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

    def get_recommended_skill_level(self) -> str:
        """Get human-readable skill requirements"""
        return {
            self.GREEN: "Beginner",
            self.BLUE: "Intermediate",
            self.BLACK: "Advanced",
            self.DOUBLE_BLACK: "Expert"
        }[self]

@dataclass
class FeatureRelationships:
    """Defines how features can interact within segments"""
    compatible_combinations: Dict[TrailFeatureType, List[TrailFeatureType]] = field(
        default_factory=lambda: {
            TrailFeatureType.ROLLER: [TrailFeatureType.DROP],
            TrailFeatureType.FLOW_DESCENT: [TrailFeatureType.SWITCHBACK],
            TrailFeatureType.TECHNICAL_DESCENT: [TrailFeatureType.DROP, TrailFeatureType.KICKER]
        }
    )
    
    mutually_exclusive: List[Tuple[TrailFeatureType, TrailFeatureType]] = field(
        default_factory=lambda: [
            (TrailFeatureType.ROLLER, TrailFeatureType.SWITCHBACK),
            (TrailFeatureType.TECHNICAL_ASCENT, TrailFeatureType.KICKER)
        ]
    )

    def can_coexist(self, feature1: TrailFeatureType, feature2: TrailFeatureType) -> bool:
        """Check if two features can exist in the same segment"""
        if (feature1, feature2) in self.mutually_exclusive:
            return False
        if (feature2, feature1) in self.mutually_exclusive:
            return False
            
        return (feature2 in self.compatible_combinations.get(feature1, []) or
                feature1 in self.compatible_combinations.get(feature2, []))
    
    def validate_segment_compatibility(self, 
                                     primary_feature: TrailFeatureType,
                                     secondary_features: List[TrailFeatureType],
                                     gradient_type: GradientSegmentType) -> bool:
        """Validate if features can coexist in a segment with given gradient"""
        if primary_feature is None:
            return True
            
        # Check gradient compatibility
        if not all(f.is_compatible_with(gradient_type) for f in [primary_feature] + secondary_features):
            return False
            
        # Check feature inter-compatibility
        for feature in secondary_features:
            if not self.can_coexist(primary_feature, feature):
                return False
                
        return True

# Predefined feature relationships
DEFAULT_FEATURE_RELATIONSHIPS = FeatureRelationships()