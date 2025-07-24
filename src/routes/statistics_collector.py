from enum import Enum, auto
from typing import Any, List, Dict, Optional
import numpy as np

from scipy.signal import find_peaks, savgol_filter

from src.routes.route_processor import ProcessedRoute
from src.routes.route import GeoPoint
from src.routes.track import Track, TrackAnalysis
from src.ui.map_helpers import print_step

MIN_SEGMENT_LENGTH = 50  # meters (minimum length for a meaningful segment)
MIN_STEEP_LENGTH = 10    # meters (allow shorter segments for very steep sections)

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
    """Enhanced GeoPoint with gradient, segment data, and baseline elevation"""
    def __init__(self, lat: float, lon: float, elevation: float, 
                 distance_from_origin: float, gradient: Optional[float] = None,
                 segment_type: Optional[ElevationSegmentType] = None,
                 baseline_elevation: Optional[float] = None):
        super().__init__(lat, lon, elevation, distance_from_origin)
        self.gradient = gradient
        self.segment_type = segment_type
        self.baseline_elevation = baseline_elevation
        self.oscillation = (elevation - baseline_elevation) if baseline_elevation is not None else None

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            'gradient': self.gradient,
            'segment_type': self.segment_type.name if self.segment_type else None,
            'baseline_elevation': self.baseline_elevation,
            'oscillation': self.oscillation
        })
        return base

class StatisticsCollector:
    def __init__(self):
        self._gradient_window = 3
        # --- New Parameters for Roller Detection ---
        self.ROLLER_BASELINE_WINDOW_LENGTH = 51  # Window length for Savitzky-Golay filter (must be odd)
        self.ROLLER_BASELINE_POLYORDER = 3       # Polynomial order for Savitzky-Golay filter
        self.ROLLER_MIN_OSCILLATIONS = 3         # Minimum number of peaks/valleys to qualify as roller
        self.ROLLER_MIN_LENGTH = 30              # Minimum length (meters) for a roller section
        self.ROLLER_MAX_LENGTH = 500             # Maximum length (meters) for a roller section
        self.ROLLER_MIN_AMPLITUDE = 2.0          # Minimum average peak-to-peak amplitude (meters)
        # --- End New Parameters ---
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
        
        # Get elevation and baseline interpolators
        elev_interp = proute.interpolators['ele']
        baseline_interp = proute.interpolators.get('baseline')
        
        # Calculate elevations and baseline
        elevations = elev_interp(t_values)
        baseline = baseline_interp(t_values) if baseline_interp else None
        
        # Calculate gradients
        if hasattr(elev_interp, 'derivative'):
            points = proute.smooth_points
            gradients = np.zeros_like(points)
            for i in range(1, len(points)):
                Δelev = points[i].elevation - points[i-1].elevation
                Δdist = points[i].distance_from_origin - points[i-1].distance_from_origin
                gradients[i] = Δelev / Δdist if Δdist > 0 else 0
        else:
            distances = [p.distance_from_origin for p in proute.smooth_points]
            gradients = np.gradient(elevations, distances)

        return [
            StaticProfilePoint(
                p.lat, p.lon, p.elevation, p.distance_from_origin,
                gradient=float(gradients[i]),
                baseline_elevation=float(baseline[i]) if baseline is not None else None
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

        # Merge segments (keep your fixed logic)
        merged_segments = [segments[0]]
        for current in segments[1:]:
            last = merged_segments[-1]
            if (current.segment_type == last.segment_type or 
                self._is_transitional(last.segment_type, current.segment_type)):
                merged = ElevationSegment(
                    start_idx=last.start_index,
                    seg_type=last.segment_type,
                    gradient=last.gradients[0],
                    distance=last.distances[0]
                )
                merged.gradients = last.gradients + current.gradients
                merged.distances = last.distances + current.distances
                merged.end_index = current.end_index
                merged_segments[-1] = merged
            else:
                merged_segments.append(current)

        # Extract profile points for oscillation analysis
        profile_points = []
        for seg in merged_segments:
            for i in range(len(seg.gradients)):
                profile_points.append(
                    StaticProfilePoint(
                        lat=0, lon=0,  # Placeholder (values unused)
                        elevation=seg.gradients[i],  # Mock elevation
                        distance_from_origin=seg.distances[i],
                        gradient=seg.gradients[i]
                    )
                )

        # Detect oscillations
        sign_changes = self._find_gradient_sign_changes(profile_points)
        oscillations = self._group_oscillations(profile_points, sign_changes)

        # Identify valid rollers (adjust thresholds as needed)
        roller_segments = []
        for osc in oscillations:
            if 5 <= osc["length"] <= 100 and abs(osc["avg_gradient"]) >= 0.05:
                roller_segments.append(
                    ElevationSegment(
                        start_idx=osc["start_idx"],
                        seg_type=ElevationSegmentType.ROLLER,
                        gradient=osc["avg_gradient"],
                        distance=profile_points[osc["start_idx"]].distance_from_origin
                    )
                )

        # Combine rollers with non-roller segments
        final_segments = [s for s in merged_segments if not self._is_roller_candidate(s)]
        final_segments.extend(roller_segments)
        return sorted(final_segments, key=lambda x: x.start_index)

    def _is_roller_candidate(self, segment: ElevationSegment) -> bool:
        """Check if a segment is short enough to be part of a roller."""
        return segment.length() < 100 and segment.segment_type in (
            ElevationSegmentType.ASCENT, 
            ElevationSegmentType.DESCENT
        )

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
    
    def _find_gradient_sign_changes(self, profile: List[StaticProfilePoint]) -> List[int]:
        """Detect indices where gradient changes sign (ascend <-> descend)."""
        sign_changes = []
        for i in range(1, len(profile)):
            if profile[i].gradient is None or profile[i-1].gradient is None:
                continue
            if (profile[i].gradient * profile[i-1].gradient) < 0:  # Sign change
                sign_changes.append(i)
        return sign_changes

    def _group_oscillations(self, profile: List[StaticProfilePoint], sign_changes: List[int]) -> List[Dict[str, Any]]:
        """Group sign changes into candidate roller sections."""
        oscillations = []
        for i in range(len(sign_changes) - 1):
            start_idx = sign_changes[i]
            end_idx = sign_changes[i + 1]
            length = profile[end_idx].distance_from_origin - profile[start_idx].distance_from_origin
            avg_gradient = np.mean([p.gradient for p in profile[start_idx:end_idx] if p.gradient is not None])
            
            oscillations.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "length": length,
                "avg_gradient": avg_gradient,
            })
            
        return oscillations