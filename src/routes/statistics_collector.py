from enum import Enum, auto
from typing import Any, List, Dict, Optional
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

class ElevationSegment:
    """Represents a continuous segment of similar elevation characteristics"""
    def __init__(self, start_idx: int, seg_type: ElevationSegmentType, 
                 gradient: float, distance: float):
        self.start_index = start_idx
        self.end_index = start_idx
        self.segment_type = seg_type
        self.distances = [distance]
        self.gradients = [gradient]
        
    def length(self) -> float:
        """Calculate segment length in meters"""
        return self.distances[-1] - self.distances[0]

    def avg_gradient(self) -> float:
        """Calculate average gradient for the segment"""
        return np.mean(self.gradients)

    def max_gradient(self) -> float:
        """Calculate maximum gradient for the segment"""
        return max(self.gradients) if self.segment_type in [
            ElevationSegmentType.ASCENT, ElevationSegmentType.STEEP_ASCENT
        ] else min(self.gradients)

    def to_dict(self) -> dict:
        """Convert segment to dictionary representation"""
        return {
            'start_index': self.start_index,
            'end_index': self.end_index,
            'segment_type': self.segment_type.name,
            'avg_gradient': self.avg_gradient(),
            'max_gradient': self.max_gradient(),
            'length': self.length(),
            'start_distance': self.distances[0],
            'end_distance': self.distances[-1]
        }

class RouteDifficulty(Enum):
    GREEN = auto()      # Easy
    BLUE = auto()       # Intermediate  
    BLACK = auto()      # Difficult
    DOUBLE_BLACK = auto() # Expert

class StaticProfilePoint(GeoPoint):    
    
    """Enhanced GeoPoint with gradient and segment data"""
    def __init__(self, lat: float, lon: float, elevation: float, 
                 elevation_baseline : float, distance_from_origin: float, gradient: Optional[float] = None,
                 segment_type: Optional[ElevationSegmentType] = None):
        super().__init__(lat, lon, elevation, distance_from_origin)
        self.gradient = gradient
        self.segment_type = segment_type
        self.elevation_baseline = elevation_baseline

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
                             associated_tracks: List[Track]) -> Dict[str, Any]:
        static_profile = self._create_static_profile(proute)
        analysed_segments = self._analyze_segments(static_profile)
        
        return {
            'static': static_profile,
            'dynamic': associated_tracks[0].analysis,
            'segments': [s.to_dict() for s in analysed_segments],
            'difficulty': self._determine_difficulty(analysed_segments).name
        }

    def _create_static_profile(self, proute: ProcessedRoute) -> List[StaticProfilePoint]:
        t_values = np.linspace(0, 1, len(proute.smooth_points))
        elev_interp = proute.interpolators['ele']
        
        if hasattr(elev_interp, 'derivative'):
            points = proute.smooth_points
            gradients = np.zeros_like(points)
            for i in range(1, len(points)):
                Δelev = points[i].elevation - points[i-1].elevation
                Δdist = points[i].distance_from_origin - points[i-1].distance_from_origin
                gradients[i] = Δelev / Δdist if Δdist > 0 else 0
        else:
            elevations = elev_interp(t_values)
            distances = [p.distance_from_origin for p in proute.smooth_points]
            gradients = np.gradient(elevations, distances)

        return [
            StaticProfilePoint(
                p.lat, p.lon, p.elevation, proute.baseline.points[i], p.distance_from_origin,
                gradient=float(gradients[i])
            )
            for i, p in enumerate(proute.smooth_points)
        ]
        
    def _analyze_segments(self, profile: List[StaticProfilePoint]) -> List[ElevationSegment]:
        segments = []
        current_segment = None

        for i, point in enumerate(profile):
            if point.gradient is None:
                continue

            current_type = self._classify_gradient(point.gradient)

            if current_segment is None:
                current_segment = ElevationSegment(i, current_type, 
                                                point.gradient, 
                                                point.distance_from_origin)
            else:
                should_continue = (
                    current_type == current_segment.segment_type or
                    (point.distance_from_origin - current_segment.distances[-1]) < 5 or
                    self._is_transitional(current_segment.segment_type, current_type)
                )
                
                if should_continue:
                    current_segment.gradients.append(point.gradient)
                    current_segment.distances.append(point.distance_from_origin)
                    current_segment.end_index = i
                else:
                    if self._validate_segment(current_segment):
                        segments.append(current_segment)
                    current_segment = ElevationSegment(i, current_type, 
                                                    point.gradient, 
                                                    point.distance_from_origin)

        if current_segment and self._validate_segment(current_segment):
            segments.append(current_segment)

        return self._post_process_segments(segments)

    def _post_process_segments(self, segments: List[ElevationSegment]) -> List[ElevationSegment]:
        if not segments:
            return []

        # 1. Merge similar adjacent segments
        merged_segments = [segments[0]]
        for current in segments[1:]:
            last = merged_segments[-1]
            
            if (current.segment_type == last.segment_type or
                self._is_transitional(last.segment_type, current.segment_type)):
                
                # *** FIX START ***
                # The original code passed lists to the ElevationSegment constructor,
                # which expects float values, causing a nested-list bug.
                # The fix is to initialize a new segment correctly using the first
                # data point, then overwrite its lists with the merged data.

                # Initialize a new segment with the starting point of the 'last' segment.
                merged = ElevationSegment(
                    start_idx=last.start_index,
                    seg_type=last.segment_type,
                    gradient=last.gradients[0],
                    distance=last.distances[0]
                )

                # Overwrite the lists with the combined data from 'last' and 'current' segments.
                merged.gradients = last.gradients + current.gradients
                merged.distances = last.distances + current.distances
                merged.end_index = current.end_index
                
                # Replace the 'last' segment in the list with the new 'merged' segment.
                merged_segments[-1] = merged
                # *** FIX END ***
            else:
                merged_segments.append(current)

        # 2. Identify roller sections
        final_segments = []
        roller_candidate = []
        
        for seg in merged_segments:
            if (seg.length() < 100 and 
                seg.segment_type in (ElevationSegmentType.ASCENT, ElevationSegmentType.DESCENT)):
                roller_candidate.append(seg)
            else:
                if len(roller_candidate) >= ROLLER_THRESHOLD:
                    final_segments.append(self._create_roller_segment(roller_candidate))
                roller_candidate = []
                final_segments.append(seg)
        
        if len(roller_candidate) >= ROLLER_THRESHOLD:
            final_segments.append(self._create_roller_segment(roller_candidate))
        
        return final_segments

    def _create_roller_segment(self, segments: List[ElevationSegment]) -> ElevationSegment:
        """Combine multiple short ascent/descent segments into one roller segment"""
        all_gradients = []
        all_distances = []
        for seg in segments:
            all_gradients.extend(seg.gradients)
            all_distances.extend(seg.distances)
        
        # *** FIX START ***
        # The original code passed lists (all_gradients, all_distances)
        # to the ElevationSegment constructor, which expects float values.
        # The fix is to initialize the segment with the first point's data,
        # then overwrite the gradient and distance lists with the combined data.
        roller = ElevationSegment(
            start_idx=segments[0].start_index,
            seg_type=ElevationSegmentType.ROLLER,
            gradient=segments[0].gradients[0],
            distance=segments[0].distances[0]
        )
        roller.gradients = all_gradients
        roller.distances = all_distances
        roller.end_index = segments[-1].end_index
        # *** FIX END ***
        return roller

    def _validate_segment(self, segment: ElevationSegment) -> bool:
        min_length = (MIN_STEEP_LENGTH if segment.segment_type in 
                     (ElevationSegmentType.STEEP_ASCENT, ElevationSegmentType.STEEP_DESCENT)
                     else MIN_SEGMENT_LENGTH)
        return segment.length() >= min_length

    def _is_transitional(self, type1: ElevationSegmentType, type2: ElevationSegmentType) -> bool:
        transitional_pairs = [
            (ElevationSegmentType.ASCENT, ElevationSegmentType.STEEP_ASCENT),
            (ElevationSegmentType.DESCENT, ElevationSegmentType.STEEP_DESCENT),
            (ElevationSegmentType.FLAT, ElevationSegmentType.ASCENT),
            (ElevationSegmentType.FLAT, ElevationSegmentType.DESCENT)
        ]
        return (type1, type2) in transitional_pairs or (type2, type1) in transitional_pairs
    
    def _determine_difficulty(self, segments: List[ElevationSegment]) -> RouteDifficulty:
        if not segments:
            return RouteDifficulty.GREEN
        
        total_climb = sum(
            s.length() * s.avg_gradient() 
            for s in segments 
            if s.avg_gradient() > 0 and s.length() > 50
        )
        
        steep_segments = [
            s for s in segments 
            if s.segment_type in (ElevationSegmentType.STEEP_ASCENT, ElevationSegmentType.STEEP_DESCENT)
            and s.length() > 20
        ]
        
        sustained_steep = any(
            s.length() > 100 and abs(s.avg_gradient()) > 0.15 
            for s in segments
        )
        
        if sustained_steep or any(s.max_gradient() > 0.25 for s in segments):
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