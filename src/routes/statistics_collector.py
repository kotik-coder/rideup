from enum import Enum, auto
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

from src.routes.route_processor import ProcessedRoute
from src.routes.route import GeoPoint
from src.routes.track import Track, TrackAnalysis
from src.ui.map_helpers import print_step

MIN_SEGMENT_LENGTH = 50  # meters (minimum length for a meaningful segment)
MIN_STEEP_LENGTH = 10    # meters (allow shorter segments for very steep sections)
ROLLER_THRESHOLD = 3     # number of oscillations to qualify as roller section

class ElevationSegmentType(Enum):
    ASCENT = auto()
    DESCENT = auto()
    STEEP_ASCENT = auto()    # >10% gradient
    STEEP_DESCENT = auto()   # <-10% gradient
    ROLLER = auto()          # Frequent elevation changes
    SWITCHBACK = auto()      # Tight turns with elevation
    FLAT = auto()            # <1% gradient

@dataclass
class ElevationSegment:
    """Represents a continuous segment of similar elevation characteristics"""
    start_index: int
    end_index: int
    segment_type: ElevationSegmentType
    avg_gradient: float
    max_gradient: float
    length: float  # in meters
    start_distance: float  # meters from route start
    end_distance: float  # meters from route start

    def to_dict(self) -> dict:
        return {
            'start_index': self.start_index,
            'end_index': self.end_index,
            'type': self.segment_type.name,
            'avg_gradient': self.avg_gradient,
            'max_gradient': self.max_gradient,
            'length': self.length,
            'start_distance': self.start_distance,
            'end_distance': self.end_distance
        }

class RouteDifficulty(Enum):
    GREEN = auto()      # Easy
    BLUE = auto()       # Intermediate  
    BLACK = auto()      # Difficult
    DOUBLE_BLACK = auto() # Expert

@dataclass
class StaticProfilePoint(GeoPoint):
    """Enhanced GeoPoint with gradient and segment data"""
    gradient: Optional[float] = None       # Slope (decimal)
    segment_type: Optional[ElevationSegmentType] = None

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            'gradient': self.gradient,
            'segment_type': self.segment_type.name if self.segment_type else None
        })
        return base


class StatisticsCollector:
    def __init__(self):
        self._gradient_window = 3
        print_step("StatisticsCollector", "Initialized with ProcessedRoute integration")

    def generate_route_profiles(self, 
                             proute: ProcessedRoute,
                             associated_tracks: List[Track]) -> Dict[str, List]:
        """
        Generates all route profiles using interpolators from ProcessedRoute
        """
        static_profile = self._create_static_profile(proute)
        return {
            'static': static_profile,
            'dynamic': associated_tracks[0].analysis,
            'segments': self._analyze_segments(static_profile),
            'difficulty': self._determine_difficulty(
                self._analyze_segments(static_profile)
            ).name
        }

    def _create_static_profile(self, proute: ProcessedRoute) -> List[StaticProfilePoint]:
        """Creates static profile using stored interpolators"""

        t_values = np.linspace(0, 1, len(proute.smooth_points))
        elev_interp = proute.interpolators['ele']
        
        # Gradient calculation that works with any scipy interpolator
        if hasattr(elev_interp, 'derivative'):  # Akima case
            gradients = elev_interp.derivative()(t_values)
        else:  # Linear or other interpolation
            elevations = elev_interp(t_values)
            distances = [p.distance_from_origin for p in proute.smooth_points]
            gradients = np.gradient(elevations, distances)

        return [
            StaticProfilePoint(
                lat=p.lat,
                lon=p.lon,
                elevation=p.elevation,
                distance_from_origin=p.distance_from_origin,
                gradient=float(gradients[i])
            )
            for i, p in enumerate(proute.smooth_points)
        ]

    def _analyze_segments(self, profile: List[StaticProfilePoint]) -> List[ElevationSegment]:
        segments = []
        current_segment = None
        gradient_buffer = []
        distance_buffer = []

        for i, point in enumerate(profile):
            if point.gradient is None:
                continue

            current_gradient = point.gradient
            current_distance = point.distance_from_origin

            # Classify the current point
            current_type = self._classify_gradient(current_gradient)

            if current_segment is None:
                # Start new segment
                current_segment = {
                    'start_idx': i,
                    'type': current_type,
                    'gradients': [current_gradient],
                    'distances': [current_distance],
                    'start_dist': current_distance
                }
            else:
                # Check if we should continue or break the segment
                should_continue = (
                    current_type == current_segment['type'] or  # Same type
                    (current_distance - current_segment['distances'][-1]) < 5 or  # Very close points
                    (self._is_transitional(current_segment['type'], current_type))  # Transitional types
                )
                if should_continue:
                    current_segment['gradients'].append(current_gradient)
                    current_segment['distances'].append(current_distance)
                else:
                    # Finalize current segment if it meets length requirements
                    segment_length = current_distance - current_segment['start_dist']
                    min_required = (MIN_STEEP_LENGTH if current_segment['type'] in 
                                (ElevationSegmentType.STEEP_ASCENT, ElevationSegmentType.STEEP_DESCENT)
                                else MIN_SEGMENT_LENGTH)

                    if segment_length >= min_required:
                        segments.append(self._create_segment_object(current_segment, i-1, profile))
                    
                    # Start new segment
                    current_segment = {
                        'start_idx': i,
                        'type': current_type,
                        'gradients': [current_gradient],
                        'distances': [current_distance],
                        'start_dist': current_distance
                    }

        # Handle the final segment
        if current_segment:
            segment_length = profile[-1].distance_from_origin - current_segment['start_dist']
            min_required = (MIN_STEEP_LENGTH if current_segment['type'] in 
                        (ElevationSegmentType.STEEP_ASCENT, ElevationSegmentType.STEEP_DESCENT)
                        else MIN_SEGMENT_LENGTH)
            
            if segment_length >= min_required:
                segments.append(self._create_segment_object(current_segment, len(profile)-1, profile))

        # Post-processing to merge similar adjacent segments and identify rollers
        return self._post_process_segments(segments)

    def _is_transitional(self, type1: ElevationSegmentType, type2: ElevationSegmentType) -> bool:
        """Determine if two segment types can be considered transitional"""
        transitional_pairs = [
            (ElevationSegmentType.ASCENT, ElevationSegmentType.STEEP_ASCENT),
            (ElevationSegmentType.DESCENT, ElevationSegmentType.STEEP_DESCENT),
            (ElevationSegmentType.FLAT, ElevationSegmentType.ASCENT),
            (ElevationSegmentType.FLAT, ElevationSegmentType.DESCENT)
        ]
        return (type1, type2) in transitional_pairs or (type2, type1) in transitional_pairs

    def _post_process_segments(self, segments: List[ElevationSegment]) -> List[ElevationSegment]:
        """Merge similar adjacent segments and identify roller sections"""
        if not segments:
            return []

        # 1. Merge similar adjacent segments
        merged_segments = [segments[0]]
        for current in segments[1:]:
            last = merged_segments[-1]
            
            # Conditions for merging:
            # - Same segment type
            # - Or transitional types with small length between them
            if (current.segment_type == last.segment_type or
                self._is_transitional(last.segment_type, current.segment_type)):
                
                # Create merged segment
                merged = ElevationSegment(
                    start_index=last.start_index,
                    end_index=current.end_index,
                    segment_type=last.segment_type,  # Keep the first type
                    avg_gradient=np.mean([last.avg_gradient, current.avg_gradient]),
                    max_gradient=(max(last.max_gradient, current.max_gradient) 
                                if last.segment_type in [ElevationSegmentType.ASCENT, ElevationSegmentType.STEEP_ASCENT]
                                else min(last.max_gradient, current.max_gradient)),
                    length=last.length + current.length,
                    start_distance=last.start_distance,
                    end_distance=current.end_distance
                )
                merged_segments[-1] = merged
            else:
                merged_segments.append(current)

        # 2. Identify roller coaster sections (frequent elevation changes)
        final_segments = []
        roller_candidate = []
        
        for seg in merged_segments:
            # Check if segment could be part of a roller section
            if seg.length < 100 and seg.segment_type in (ElevationSegmentType.ASCENT, 
                                                        ElevationSegmentType.DESCENT):
                roller_candidate.append(seg)
            else:
                if len(roller_candidate) >= ROLLER_THRESHOLD:
                    # Create a roller segment
                    roller_segment = self._create_roller_segment(roller_candidate)
                    final_segments.append(roller_segment)
                    roller_candidate = []
                final_segments.append(seg)
        
        # Handle any remaining roller candidate
        if len(roller_candidate) >= ROLLER_THRESHOLD:
            final_segments.append(self._create_roller_segment(roller_candidate))
        
        return final_segments

    def _create_roller_segment(self, segments: List[ElevationSegment]) -> ElevationSegment:
        """Combine multiple short ascent/descent segments into one roller segment"""
        start = segments[0].start_index
        end = segments[-1].end_index
        length = segments[-1].end_distance - segments[0].start_distance
        
        # Calculate representative gradient (average of absolute gradients)
        abs_gradients = [abs(s.avg_gradient) for s in segments]
        avg_gradient = np.mean(abs_gradients) * (1 if segments[0].avg_gradient > 0 else -1)
        
        return ElevationSegment(
            start_index=start,
            end_index=end,
            segment_type=ElevationSegmentType.ROLLER,
            avg_gradient=avg_gradient,
            max_gradient=max(s.max_gradient for s in segments),
            length=length,
            start_distance=segments[0].start_distance,
            end_distance=segments[-1].end_distance
        )

    def _create_segment_object(self, segment_data: Dict, end_idx: int, 
                             profile: List[StaticProfilePoint]) -> ElevationSegment:
        start_idx = segment_data['start_idx']
        gradients = segment_data['gradients']
        seg_type = segment_data['type']
        
        return ElevationSegment(
            start_index=start_idx,
            end_index=end_idx,
            segment_type=seg_type,
            avg_gradient=np.mean(gradients),
            max_gradient=(max(gradients) if seg_type in [
                ElevationSegmentType.ASCENT, ElevationSegmentType.STEEP_ASCENT
            ] else min(gradients)),
            length=profile[end_idx].distance_from_origin - segment_data['start_dist'],
            start_distance=profile[start_idx].distance_from_origin,
            end_distance=profile[end_idx].distance_from_origin
        )

    def _determine_difficulty(self, segments: List[ElevationSegment]) -> RouteDifficulty:
        if not segments:
            return RouteDifficulty.GREEN

        # Calculate metrics considering sustained segments
        total_climb = sum(s.length * s.avg_gradient for s in segments 
                        if s.avg_gradient > 0 and s.length > 50)
        total_descent = abs(sum(s.length * s.avg_gradient for s in segments 
                            if s.avg_gradient < 0 and s.length > 50))
        
        # Count only significant steep segments
        steep_segments = [s for s in segments 
                        if s.segment_type in (ElevationSegmentType.STEEP_ASCENT, 
                                            ElevationSegmentType.STEEP_DESCENT)
                        and s.length > 20]
        
        # Check for sustained steep sections
        sustained_steep = any(s.length > 100 and abs(s.avg_gradient) > 0.15 
                            for s in segments)
        
        if sustained_steep or any(s.max_gradient > 0.25 for s in segments):
            return RouteDifficulty.DOUBLE_BLACK
        elif len(steep_segments) > 3 or total_climb > 800:
            return RouteDifficulty.BLACK
        elif len(steep_segments) > 1 or total_climb > 300:
            return RouteDifficulty.BLUE
        return RouteDifficulty.GREEN
    
    def _classify_gradient(self, gradient: float) -> ElevationSegmentType:
        if gradient > 0.1: return ElevationSegmentType.STEEP_ASCENT
        if gradient < -0.1: return ElevationSegmentType.STEEP_DESCENT
        if gradient > 0.01: return ElevationSegmentType.ASCENT
        if gradient < -0.01: return ElevationSegmentType.DESCENT
        return ElevationSegmentType.FLAT