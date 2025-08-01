
from enum import Enum
from tkinter import FLAT
from typing import Optional

import numpy as np

from src.routes.route_processor import ProcessedRoute

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
            if segment_type.min_grad is not None and segment_type.max_grad is not None:
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
    ROLLER = (None, None)
    SWITCHBACK = (None, None) 
    TECHNICAL_DESCENT = (0.15, None)  # Min 15% gradient
    FLOW_DESCENT = (None, None)
    DROP_SECTION = (0.15, 15)  # Min 15% gradient AND max 15m length

    def __init__(self, min_gradient: Optional[float], max_length: Optional[float]):
        self.min_gradient = min_gradient
        self.max_length = max_length

    @classmethod
    def is_technical(cls, gradient: float, length: float) -> Optional['TrailFeatureType']:
        """Returns the appropriate feature type if technical thresholds are met"""
        for feature in cls:
            if (feature.min_gradient is not None and abs(gradient) >= feature.min_gradient and
                (feature.max_length is None or length <= feature.max_length)):
                return feature
        return None

class ElevationSegment:
    
    MIN_SEGMENT_LENGTH = 50         # metres
    MIN_STEEP_LENGTH = 10           # metres
    TECHNICAL_LENGTH_THRESHOLD = 15 # metres
    
    """Represents a continuous segment of similar elevation characteristics"""
    def __init__(self, 
                 start_idx: int, 
                 gradient_type: GradientSegmentType,
                 feature_type: Optional[TrailFeatureType],
                 gradient: float, 
                 distance: float,
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
        self.baseline_gradient = 0
        self.riding_context = "GENERIC"  # Also good practice to initialize this        
    
    def classify_technical_feature(self):
        
        feature_type = TrailFeatureType.is_technical(
            gradient=max(abs(g) for g in self.gradients),
            length=self.length()
        )
        
        if feature_type:
            self.feature_type = feature_type
        
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
    
    def refine(self, baseline_gradients: np.ndarray):
        """Applies baseline refinement to a single segment"""
        seg_gradients = baseline_gradients[self.start_index:self.end_index+1]
        avg_baseline_grad = np.mean(seg_gradients)
        baseline_type = GradientSegmentType.from_gradient(avg_baseline_grad)

        # Only override gradient type for sustained features
        if baseline_type is not FLAT:
            self.gradient_type = baseline_type         
    
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
        min_length = (self.MIN_STEEP_LENGTH if self.gradient_type in 
                     (GradientSegmentType.STEEP_ASCENT, GradientSegmentType.STEEP_DESCENT)
                     else self.MIN_SEGMENT_LENGTH)
        return self.length() >= min_length
    
    def classify_sustained_feature(self, proute: ProcessedRoute):
        """Classifies sustained features using baseline interpolation"""
        # Get start and end distances
        
        dist = lambda i : proute.smooth_points[i].distance_from_origin 
        
        start_dist     = dist(self.start_index)
        end_dist       = dist(self.end_index)
        total_distance = proute.total_distance()
        
        # Calculate baseline gradient
        t_start = start_dist / total_distance
        t_end = end_dist / total_distance
        elev_start = proute.baseline.get_baseline_elevation(t_start)
        elev_end   = proute.baseline.get_baseline_elevation(t_end)
        segment_length = end_dist - start_dist
        avg_baseline_grad = (elev_end - elev_start) / segment_length if segment_length > 0 else 0
        
        base_type = GradientSegmentType.from_gradient(avg_baseline_grad)
        
        # Technical overlay check
        max_raw_grad = max(abs(g) for g in self.gradients)
        if max_raw_grad > TrailFeatureType.TECHNICAL_DESCENT.min_gradient:
            if base_type in (GradientSegmentType.DESCENT, GradientSegmentType.STEEP_DESCENT):
                feature_type = TrailFeatureType.TECHNICAL_DESCENT
            else:
                feature_type = None
        else:
            feature_type = None
        
        self.gradient_type = base_type
        self.feature_type  = feature_type
       
    def is_sustained_feature(self, proute: ProcessedRoute) -> bool:
        """Identifies baseline-dominant features using baseline interpolation"""
        # Get start and end distances
        elev_start, elev_end = proute.get_baseline_elevations_on(self.start_index, self.end_index)
        
        seg_len = self.length()
        
        if seg_len < 1e-16:
            return False
        
        avg_baseline_grad = (elev_end - elev_start) / seg_len
        
        return (abs(avg_baseline_grad) > 0.05 
                and seg_len > 50)
        
    def is_roller_candidate(self) -> bool:
        """Determines if a segment qualifies for roller analysis using both gradient
        patterns and baseline residuals."""
        
        # Basic length and type requirements
        if self.length() < 15:  # Minimum 15m for consideration
            return False
            
        if self.gradient_type not in (GradientSegmentType.ASCENT, 
                                     GradientSegmentType.DESCENT,
                                     GradientSegmentType.STEEP_DESCENT):
            return False
        
        # Gradient oscillation check (at least 3 significant reversals)
        gradient_changes = np.diff(np.sign(self.gradients))
        reversal_count   = np.sum(np.abs(gradient_changes) > 0)
        return reversal_count >= 2  # 2 changes = 3 monotonic sections
        
    def length(self) -> float:
        """Calculate segment length in meters"""
        return self.distances[-1] - self.distances[0]

    def avg_gradient(self) -> float:
        """Calculate average gradient for the segment"""
        return np.mean(self.gradients)

    def max_gradient(self) -> float:
        """Calculate maximum gradient for the segment"""
        return max(self.gradients) if self.gradient_type in [
            GradientSegmentType.ASCENT, GradientSegmentType.STEEP_ASCENT
        ] else min(self.gradients)

    def to_dict(self) -> dict:
        """Convert segment to dictionary representation"""
        return {
            'start_index': self.start_index,
            'end_index': self.end_index,
            'segment_type': self.gradient_type.name,
            'avg_gradient': self.avg_gradient(),
            'max_gradient': self.max_gradient(),
            'length': self.length(),
            'start_distance': self.distances[0],
            'end_distance': self.distances[-1]
        }

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