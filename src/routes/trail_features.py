from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Tuple
from src.routes.spot import RatingSystem

class GradientSegmentType(Enum):
    """Classification of gradient segments using RatingSystem thresholds"""
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
        """Classify gradient using RatingSystem thresholds"""
        for segment_type in cls:
            min_grad, max_grad = segment_type.get_thresholds(spot_system)
            if min_grad <= gradient < max_grad:
                return segment_type
        return cls.FLAT

    def is_transitional_to(self, other: 'GradientSegmentType') -> bool:
        """Check if transition between gradient types is allowed"""
        transitional_pairs = {
            (self.ASCENT, self.STEEP_ASCENT),
            (self.DESCENT, self.STEEP_DESCENT),
            (self.FLAT, self.ASCENT),
            (self.FLAT, self.DESCENT)
        }
        return (self, other) in transitional_pairs or (other, self) in transitional_pairs
    
class TrailFeatureType(Enum):
    """Types of trail features with RatingSystem configuration"""
    ROLLER = auto()
    SWITCHBACK = auto()
    TECHNICAL_DESCENT = auto()
    TECHNICAL_ASCENT = auto()
    FLOW_DESCENT = auto()
    KICKER = auto()
    DROP = auto()

    def get_config(self, rating_system: RatingSystem) -> Dict[str, Any]:
        """Get feature parameters from RatingSystem"""
        return rating_system.get_feature_config(self.name)

    def is_compatible_with(self, gradient_type: GradientSegmentType, rating_system: RatingSystem) -> bool:
        """Check if feature can exist on given gradient type"""
        return rating_system.is_feature_compatible(self.name, gradient_type.name)

@dataclass
class ShortFeature:
    """Localized trail feature with validation against RatingSystem"""
    feature_type: TrailFeatureType
    gradient_type: GradientSegmentType
    start_index: int
    end_index: int
    max_gradient: float
    length: float

    def validate(self, rating_system: RatingSystem) -> bool:
        """Validate feature against RatingSystem parameters"""
        config = self.feature_type.get_config(rating_system)
        return (config['min_length'] <= self.length <= config['max_length'] and
                config['gradient_range'][0] <= self.max_gradient <= config['gradient_range'][1])

class RouteDifficulty(Enum):
    """Route difficulty classification using RatingSystem thresholds"""
    GREEN = "GREEN"
    BLUE = "BLUE"
    BLACK = "BLACK"
    DOUBLE_BLACK = "DOUBLE_BLACK"

    def get_thresholds(self, spot_system: RatingSystem) -> Tuple[float, float]:
        """Get difficulty thresholds from RatingSystem"""
        return spot_system.difficulty_thresholds[self.value]

    @classmethod
    def from_score(cls, score: float, spot_system: RatingSystem) -> 'RouteDifficulty':
        """Classify difficulty score using RatingSystem thresholds"""
        for difficulty in cls:
            min_thresh, max_thresh = difficulty.get_thresholds(spot_system)
            if min_thresh <= score < max_thresh:
                return difficulty
        return cls.GREEN