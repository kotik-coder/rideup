
from enum import Enum, auto
from typing import List, Optional

import numpy as np

class GradientSegmentType(Enum):
    """Segment types defined purely by elevation gradient"""
    ASCENT        = (0.01, 0.10)
    DESCENT       = (-0.10, -0.01)
    STEEP_ASCENT  = (0.10, float('inf'))
    STEEP_DESCENT = (-float('inf'), -0.10)
    FLAT          = (-0.01, 0.01)

    def __init__(self, min_grad: float, max_grad: float):
        self.min_grad = min_grad
        self.max_grad = max_grad

    @classmethod
    def from_gradient(cls, gradient: float) -> 'GradientSegmentType':
        """Classify a gradient into segment type"""
        for segment_type in cls:
            if segment_type.min_grad <= gradient < segment_type.max_grad:
                return segment_type
        return cls.FLAT

    def is_transitional_to(self, other: 'GradientSegmentType') -> bool:
        """Check if transition between two segment types is gradual"""
        transitional_pairs = {
            (self.ASCENT, self.STEEP_ASCENT),
            (self.DESCENT, self.STEEP_DESCENT),
            (self.FLAT, self.ASCENT),
            (self.FLAT, self.DESCENT)
        }
        return (self, other) in transitional_pairs or (other, self) in transitional_pairs
    
class TrailFeatureType(Enum):
    """Special trail features with their gradient thresholds"""
    ROLLER = auto()
    SWITCHBACK = auto()
    TECHNICAL_DESCENT = auto()
    TECHNICAL_ASCENT = auto()
    FLOW_DESCENT = auto()
    DROP_SECTION = auto()
    # New feature types
    SHORT_ASCENT = auto()       # Rapid one-time ascent
    SHORT_DESCENT = auto()      # Rapid one-time descent
    STEP_UP = auto()            # Very short, steep ascent
    STEP_DOWN = auto()          # Very short, steep descent

class ElevationSegment:
            
    # Segment length thresholds
    MIN_SEGMENT_LENGTH = 50         # metres
    MIN_STEEP_LENGTH = 10           # metres
    TECHNICAL_LENGTH_THRESHOLD = 15  # metres
    
    # Feature classification parameters
    STEP_FEATURE_MAX_LENGTH = 15        # meters
    SHORT_FEATURE_MIN_LENGTH = 15       # meters
    SHORT_FEATURE_MAX_LENGTH = 50       # meters
    SHORT_ASCENT_MIN_GRADIENT = 0.08    # 8%
    SHORT_DESCENT_MAX_GRADIENT = -0.08  # -8%
    
    # Technical section parameters
    TECHNICAL_GRADIENT_STD_THRESHOLD = 0.05  # Standard deviation threshold
    TECHNICAL_AVG_GRADE_THRESHOLD = 0.15     # 15% average grade
    
    # Elevation change parameters
    MIN_ELEVATION_CHANGE = 5           # meters minimum for significant features
    
    # Wavelength analysis parameters
    WAVELENGTH_CLUSTERING_EPS = 0.5    # DBSCAN epsilon parameter
    WAVELENGTH_MATCH_TOLERANCE = 0.3   # 30% tolerance for wavelength matching
    FLOW_WAVELENGTH_MIN = 10           # meters
    FLOW_WAVELENGTH_MAX = 50           # meters
    
    start_index : int
    end_index : int
    gradient_type : GradientSegmentType
    feature_type : TrailFeatureType
    wavelengths : List[float]
    distances : List[float]
    gradients : List[float]
    short_features : List[TrailFeatureType]
    
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
        self.short_features = []  # New list to store short, steep features       
        
    def determine_riding_context(self, segments, index):
        """Updated to check both gradient and feature types"""
        # Check if part of sustained climb
        if (self.gradient_type in [GradientSegmentType.ASCENT, GradientSegmentType.STEEP_ASCENT] and
            any(s.gradient_type == self.gradient_type 
                for s in segments[max(0,index-2):min(len(segments),index+3)])):
            return "ENDURANCE"
            
        # Check technical descent chain
        if (self.feature_type in [TrailFeatureType.TECHNICAL_DESCENT, TrailFeatureType.DROP_SECTION] and
            any(s.feature_type in [TrailFeatureType.TECHNICAL_DESCENT, TrailFeatureType.DROP_SECTION]
                for s in segments[max(0,index-1):min(len(segments),index+2)])):
            return "TECHNICAL"
            
        # Flow section detection
        if self.feature_type == TrailFeatureType.FLOW_DESCENT:
            return "FLOW"
            
        return "GENERIC"
        
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
    
    def validate(self) -> bool:
        min_length = (self.MIN_STEEP_LENGTH if self.gradient_type in (GradientSegmentType.STEEP_ASCENT, GradientSegmentType.STEEP_DESCENT)
                     else self.MIN_SEGMENT_LENGTH)
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
    GREEN = (0, 0.3)       # Easy
    BLUE = (0.3, 0.8)      # Intermediate  
    BLACK = (0.8, 1.5)     # Difficult
    DOUBLE_BLACK = (1.5, float('inf'))  # Expert
    
    def __init__(self, min_threshold, max_threshold):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
    
    @classmethod
    def from_score(cls, score: float):
        for difficulty in cls:
            if difficulty.min_threshold <= score < difficulty.max_threshold:
                return difficulty
        return cls.GREEN
