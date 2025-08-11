
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.routes.spot import RatingSystem

class GradientSegmentType(Enum):
    """Enum that pulls thresholds from SpotSystem"""
    ASCENT = "ASCENT"
    DESCENT = "DESCENT" 
    STEEP_ASCENT = "STEEP_ASCENT"
    STEEP_DESCENT = "STEEP_DESCENT"
    FLAT = "FLAT"

    def get_thresholds(self, spot_system: RatingSystem) -> Tuple[float, float]:
        """Get thresholds directly from SpotSystem"""
        return spot_system.gradient_thresholds[self.value]

    @classmethod
    def from_gradient(cls, gradient: float, spot_system: RatingSystem) -> 'GradientSegmentType':
        """Classify gradient using dictionary lookup"""
        for segment_type in cls:
            min_grad, max_grad = segment_type.get_thresholds(spot_system)
            if min_grad <= gradient < max_grad:
                return segment_type
        return cls.FLAT

    def is_transitional_to(self, other: 'GradientSegmentType') -> bool:
        """Transition logic remains enum-based"""
        transitional_pairs = {
            (self.ASCENT, self.STEEP_ASCENT),
            (self.DESCENT, self.STEEP_DESCENT),
            (self.FLAT, self.ASCENT),
            (self.FLAT, self.DESCENT)
        }
        return (self, other) in transitional_pairs or (other, self) in transitional_pairs
    
class TrailFeatureType(Enum):
    """Trail features that reference RatingSystem for parameters"""
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
        """Check compatibility using RatingSystem"""
        return rating_system.is_feature_compatible(self.name, gradient_type.name)

@dataclass
class ShortFeature:
    """Feature that validates against RatingSystem parameters"""
    feature_type: TrailFeatureType
    gradient_type: GradientSegmentType
    start_index: int
    end_index: int
    max_gradient: float
    length: float

    def validate(self, rating_system: RatingSystem) -> bool:
        """Validate against RatingSystem parameters"""
        config = self.feature_type.get_config(rating_system)
        return (config['min_length'] <= self.length <= config['max_length'] and
                config['gradient_range'][0] <= self.max_gradient <= config['gradient_range'][1])

class ElevationSegment:
    
    start_index : int
    end_index : int
    gradient_type : GradientSegmentType
    feature_type : TrailFeatureType
    wavelengths : List[float]
    distances : List[float]
    gradients : List[float]
    short_features: List[ShortFeature]
    
    def __init__(self, 
                 start_idx: int, 
                 gradient_type: GradientSegmentType,
                 gradient: float, 
                 distance: float,
                 feature_type: Optional[TrailFeatureType] = None,
                 end_idx: int = -1
                 ):
        self.start_index = start_idx
        if(end_idx > 0): 
            self.end_index = end_idx
        else:
            self.end_index = start_idx
        self.gradient_type = gradient_type
        self.feature_type = feature_type
        self.distances = [distance]
        self.gradients = [gradient]
        self.wavelengths = []
        self.riding_context = "GENERIC"
        self.short_features = []  # store short, steep features               
        
    def get_plot_data(self) -> Dict[str, Any]:
        return {
            'start': self.start_index,
            'end': self.end_index,
            'gradient_type': self.gradient_type.name,
            'feature_type': self.feature_type.name if self.feature_type else None,
            'avg_gradient': self.avg_gradient(),
            'length': self.length()
        }

    def extend(self, 
                    end_idx: int,
                    gradient: float, 
                    distance: float,
                    feature_type: Optional[TrailFeatureType] = None):
        """Extended version that can update feature type"""
        self.gradients.append(gradient)
        self.distances.append(distance)
        self.end_index = end_idx
        
        if feature_type is not None:
            self.feature_type = feature_type

    def should_continue(self, 
                            new_gradient_type: GradientSegmentType,
                            new_feature_type: Optional[TrailFeatureType] = None) -> bool:
        """Enhanced continuation logic that considers both gradient and features"""
        gradient_ok = (new_gradient_type == self.gradient_type or 
                    self.gradient_type.is_transitional_to(new_gradient_type))
        
        feature_ok = (new_feature_type is None or 
                    new_feature_type == self.feature_type)
        
        return gradient_ok and feature_ok
        
    def validate(self, spot_system: RatingSystem) -> bool:
        min_length = (spot_system.min_steep_length 
                    if self.gradient_type in (GradientSegmentType.STEEP_ASCENT, 
                                            GradientSegmentType.STEEP_DESCENT)
                    else spot_system.min_segment_length)
                    
        # Additional validation for feature segments
        if self.feature_type:
            config = self.feature_type.get_config(spot_system)
            if not (config['min_length'] <= self.length() <= config['max_length']):
                return False
                
        return self.length() >= min_length
    
    def is_roller_candidate(self) -> bool:
        """Determines if a segment qualifies for roller analysis using both gradient
        patterns and baseline residuals."""
        
        if self.gradient_type in (GradientSegmentType.ASCENT, 
                                  GradientSegmentType.STEEP_ASCENT):
            return False
        
        # Gradient oscillation check (at least 3 significant reversals)
        gradient_changes = np.diff(np.sign(self.gradients))
        reversal_count   = np.sum(np.abs(gradient_changes) > 0)
        return reversal_count > 2 
        
    def length(self) -> float:
        """Calculate segment length in meters"""
        return self.distances[-1] - self.distances[0]

    def avg_gradient(self) -> float:
        """Calculate average gradient for the segment"""
        return np.mean(self.gradients)

    def max_gradient(self) -> float:
        """Calculate maximum gradient for the segment"""
        return max(self.gradients) if self.gradient_type in [
            GradientSegmentType.ASCENT, GradientSegmentType.STEEP_ASCENT, GradientSegmentType.FLAT
        ] else min(self.gradients)

class RouteDifficulty(Enum):
    GREEN = "GREEN"
    BLUE = "BLUE"
    BLACK = "BLACK"
    DOUBLE_BLACK = "DOUBLE_BLACK"

    def get_thresholds(self, spot_system: RatingSystem) -> Tuple[float, float]:
        """Get difficulty thresholds from SpotSystem"""
        return spot_system.difficulty_thresholds[self.value]

    @classmethod
    def from_score(cls, score: float, spot_system: RatingSystem) -> 'RouteDifficulty':
        """Classify difficulty score using SpotSystem thresholds"""
        for difficulty in cls:
            min_thresh, max_thresh = difficulty.get_thresholds(spot_system)
            if min_thresh <= score < max_thresh:
                return difficulty
        return cls.GREEN