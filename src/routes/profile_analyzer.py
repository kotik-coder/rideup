from typing import List, Optional
import numpy as np
from sklearn.cluster import DBSCAN
from src.routes.spot import RatingSystem
from src.routes.route_processor import ProcessedRoute
from src.routes.route import GeoPoint
from src.routes.trail_features import GradientSegmentType, ShortFeature, TrailFeatureType
from src.routes.trail_features import TrailFeatureType as tft
from src.routes.trail_features import GradientSegmentType as gst
from scipy.signal import argrelextrema

class StaticProfilePoint(GeoPoint):
    """Basic GeoPoint with gradient data"""
    def __init__(self, 
                 lat: float, 
                 lon: float, 
                 elevation: float, 
                 distance_from_origin: float, 
                 gradient: float,
                 baseline: float = 0,
                 baseline_gradient: float = 0):
        super().__init__(lat, lon, elevation, distance_from_origin)
        self.gradient = gradient
        self.baseline = baseline
        self.baseline_gradient = baseline_gradient

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            'gradient': self.gradient,
            'baseline': self.baseline,
            'baseline_gradient': self.baseline_gradient
        })
        return base

class Segment:
    """A continuous segment of the trail with consistent characteristics"""
    def __init__(self, 
                 start_idx: int,
                 gradient_type: GradientSegmentType,
                 points: List[StaticProfilePoint],
                 feature_type: Optional[TrailFeatureType] = None):
        self.start_index = start_idx
        self.end_index = start_idx + len(points) - 1
        self.gradient_type = gradient_type
        self.feature_type = feature_type
        self.points = points
        self.short_features = []
    
    @property
    def gradients(self) -> List[float]:
        return [p.gradient for p in self.points]
    
    @property
    def distances(self) -> List[float]:
        return [p.distance_from_origin for p in self.points]
    
    def add_point(self, point: StaticProfilePoint):
        """Add a point to the segment"""
        self.points.append(point)
        self.end_index += 1
    
    def create_subsegment(self, start_idx: int, end_idx: int) -> 'Segment':
        """Create a subsegment from this segment"""
        sub_points = self.points[start_idx - self.start_index : end_idx - self.start_index + 1]
        return Segment(
            start_idx=start_idx,
            gradient_type=self.gradient_type,
            points=sub_points,
            feature_type=self.feature_type
        )
    
    def should_continue(self, new_gradient_type: GradientSegmentType) -> bool:
        """Check if segment should continue with new gradient type"""
        return (new_gradient_type == self.gradient_type or 
                self.gradient_type.is_transitional_to(new_gradient_type))
    
    def validate(self, spot_system: RatingSystem) -> bool:
        """Validate segment against rating system requirements"""
        min_length = (spot_system.min_steep_length 
                    if self.gradient_type in (gst.STEEP_ASCENT, gst.STEEP_DESCENT)
                    else spot_system.min_segment_length)
                    
        if self.feature_type:
            config = self.feature_type.get_config(spot_system)
            if not (config['min_length'] <= self.length() <= config['max_length']):
                return False
                
        return self.length() >= min_length
    
    def is_roller_candidate(self) -> bool:
        """Check if segment qualifies for roller analysis"""
        if self.gradient_type in (gst.ASCENT, gst.STEEP_ASCENT):
            return False
        
        gradient_changes = np.diff(np.sign(self.gradients))
        reversal_count = np.sum(np.abs(gradient_changes) > 0)
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
            gst.ASCENT, gst.STEEP_ASCENT, gst.FLAT
        ] else min(self.gradients)

class Profile:
    """Top-level profile container with classification logic"""
    def __init__(self, spot_system: RatingSystem, proute: ProcessedRoute):
        self.spot_system = spot_system
        self.segments = self._build_segments(proute)
    
    def _build_segments(self, proute: ProcessedRoute) -> List[Segment]:
        """Build and classify all segments"""
        points = self._create_profile_points(proute)
        raw_segments = self._identify_baseline_segments(points)
        processed_segments = []
        
        for seg in raw_segments:
            if seg.is_roller_candidate():
                self._classify_roller(seg, proute)
            self._classify_local_features(seg)
            processed_segments.append(seg)
        
        return self._merge_similar_segments(processed_segments)
    
    def _create_profile_points(self, proute: ProcessedRoute) -> List[StaticProfilePoint]:
        """Create all profile points from processed route"""
        t_values = np.linspace(0, 1, len(proute.smooth_points))
        gradients = [proute.get_gradient(t) for t in t_values]
        t = lambda i: proute.baseline.x[i]/proute.baseline.x[-1]
        
        return [
            StaticProfilePoint(
                p.lat,
                p.lon,
                p.elevation,
                p.distance_from_origin,
                gradient=gradients[i],
                baseline=proute.baseline.y[i],
                baseline_gradient=proute.baseline.get_baseline_gradient(t(i))
            )
            for i, p in enumerate(proute.smooth_points)
        ]
    
    def _identify_baseline_segments(self, points: List[StaticProfilePoint]) -> List[Segment]:
        """Identify baseline segments from points"""
        segments = []
        current_segment = None
        
        for i, point in enumerate(points):
            current_gradient_type = gst.from_gradient(point.baseline_gradient, self.spot_system)
            
            if current_segment is None:
                current_segment = Segment(
                    start_idx=i,
                    gradient_type=current_gradient_type,
                    points=[point]
                )
            else:
                if current_segment.should_continue(current_gradient_type):
                    current_segment.add_point(point)
                else:
                    if current_segment.validate(self.spot_system):
                        segments.append(current_segment)
                    current_segment = Segment(
                        start_idx=i,
                        gradient_type=current_gradient_type,
                        points=[point]
                    )
        
        if current_segment and current_segment.validate(self.spot_system):
            segments.append(current_segment)
        
        return segments
    
    def _classify_roller(self, segment: Segment, route: ProcessedRoute):
        """Classify roller segments using wavelength analysis"""
        distances = segment.distances
        residual = route.get_oscillations(segment.start_index, segment.end_index)
        
        fft_wavelengths = self._get_fft_based_wavelengths(route.baseline.freqs, segment.length())
        extrema_wavelengths = self._get_extrema_based_wavelengths(residual, distances)
        
        matched_wavelengths = self._find_matching_wavelengths(
            fft_wavelengths, 
            extrema_wavelengths,
            tolerance=self.spot_system.wavelength_match_tolerance
        )        
        
        self._classify_by_wavelength_patterns(segment, matched_wavelengths)

    def _get_fft_based_wavelengths(self, freqs, segment_length):
        """Convert FFT frequencies to relevant wavelengths"""
        if not freqs or len(freqs) == 0:
            return []
        wavelengths = 1 / freqs
        return [w for w in wavelengths if w <= segment_length * 0.8]

    def _get_extrema_based_wavelengths(self, residual, distances):
        """Get wavelengths from local extrema with quality checks"""
        peaks = argrelextrema(residual, np.greater, order=2)[0]
        troughs = argrelextrema(residual, np.less, order=2)[0]
        extrema = np.sort(np.concatenate([peaks, troughs]))
        
        if len(extrema) < 2:
            return []
        
        raw_wavelengths = [distances[extrema[i]] - distances[extrema[i-1]] 
                         for i in range(1, len(extrema))]
        
        return self._cluster_wavelengths(raw_wavelengths, eps=self.spot_system.wavelength_clustering_eps)

    def _find_matching_wavelengths(self, fft_wavelengths, extrema_wavelengths, tolerance):
        """Find wavelengths present in both FFT and extrema analysis"""
        matches = []
        for fft_w in fft_wavelengths:
            for ext_w in extrema_wavelengths:
                if abs(fft_w - ext_w) <= tolerance * min(fft_w, ext_w):
                    matches.append((fft_w, ext_w))
        
        return list({round((fft+ext)/2, 1) for fft, ext in matches}) if matches else []

    def _classify_by_wavelength_patterns(self, segment: Segment, wavelengths: List[float]) -> None:
        """Pure wavelength-based classification"""
        if not wavelengths:
            if segment.gradient_type in [gst.DESCENT, gst.STEEP_DESCENT]:
                segment.feature_type = tft.TECHNICAL_DESCENT        
            elif segment.gradient_type in [gst.ASCENT, gst.STEEP_ASCENT]:
                segment.feature_type = tft.TECHNICAL_ASCENT
            return
        
        gradient_type_name = segment.gradient_type.name
        compatible_features = self.spot_system.get_compatible_features(gradient_type_name)
        
        for w in sorted(wavelengths):
            for feature_name in ['FLOW_DESCENT', 'ROLLER']:
                if feature_name not in compatible_features:
                    continue
                    
                feature_type = getattr(tft, feature_name)
                config = feature_type.get_config(self.spot_system)
                
                if ('wavelength_range' in config and 
                    config['wavelength_range'][0] <= w <= config['wavelength_range'][1]):
                    segment.feature_type = feature_type
                    return

    def _cluster_wavelengths(self, raw_wavelengths, eps=0.5):
        """Group similar wavelengths to find dominant patterns"""
        if not raw_wavelengths:
            return []
        
        wavelengths = np.array(raw_wavelengths).reshape(-1, 1)
        db = DBSCAN(eps=eps, min_samples=2).fit(wavelengths)
        cluster_labels = db.labels_
        
        clustered_wavelengths = []
        for label in set(cluster_labels):
            if label != -1:
                cluster_points = wavelengths[cluster_labels == label]
                clustered_wavelengths.append(float(np.median(cluster_points)))
        
        return sorted(clustered_wavelengths, reverse=True) if clustered_wavelengths else [float(np.median(wavelengths))]

    def _classify_local_features(self, segment: Segment):
        """Analyze gradient trends to identify local features"""
        if segment.feature_type:
            return
            
        gradients = np.array(segment.gradients)
        distances = np.array(segment.distances)
        avg_gradient = segment.avg_gradient()
        gradient_std = np.std(gradients)
        
        upper_threshold = avg_gradient + (2 * gradient_std)
        lower_threshold = avg_gradient - (2 * gradient_std)
        
        extreme_points = np.where(
            (gradients > upper_threshold) | (gradients < lower_threshold)
        )[0]
        
        extreme_features = []
        if len(extreme_points) > 0:
            current_start = extreme_points[0]
            for i in range(1, len(extreme_points)):
                if extreme_points[i] != extreme_points[i-1] + 1:
                    extreme_features.append((current_start, extreme_points[i-1]))
                    current_start = extreme_points[i]
            extreme_features.append((current_start, extreme_points[-1]))
        
        for start_idx, end_idx in extreme_features:
            feature_length = distances[end_idx] - distances[start_idx]
            if (feature_length >= self.spot_system.step_feature_max_length or 
                feature_length <= 0):
                continue
                
            feature_gradients = gradients[start_idx:end_idx+1]
            avg_feature_gradient = np.mean(feature_gradients)
            max_feature_gradient = (np.max(feature_gradients) if avg_feature_gradient > 0 
                                  else np.min(feature_gradients))
            
            existing = any(
                sf.start_index == segment.start_index + start_idx and
                sf.end_index == segment.start_index + end_idx
                for sf in segment.short_features
            )
            if existing:
                continue
                
            steep_ascent_min = gst.STEEP_ASCENT.get_thresholds(self.spot_system)[0]
            steep_descent_max = gst.STEEP_DESCENT.get_thresholds(self.spot_system)[1]
            
            if avg_feature_gradient > steep_ascent_min:
                ftr_type = tft.KICKER
                gradient_type = gst.STEEP_ASCENT
            elif avg_feature_gradient < steep_descent_max:
                ftr_type = tft.DROP
                gradient_type = gst.STEEP_DESCENT
            else:
                continue            

            if not self.spot_system.is_feature_compatible(ftr_type.name, gradient_type.name):
                continue

            config = ftr_type.get_config(self.spot_system)
            if not (config['min_length'] <= feature_length <= config['max_length']):
                continue
            
            segment.short_features.append(
                ShortFeature(
                    feature_type=ftr_type,
                    gradient_type=gradient_type,
                    start_index=segment.start_index + start_idx,
                    end_index=segment.start_index + end_idx,
                    max_gradient=max_feature_gradient,
                    length=feature_length
                )
            ) 
        
        if not segment.feature_type and len(set(
            (f.start_index, f.end_index) for f in segment.short_features
        )) >= 2:
            if segment.gradient_type in [gst.ASCENT, gst.STEEP_ASCENT]:
                segment.feature_type = tft.TECHNICAL_ASCENT
            elif segment.gradient_type in [gst.DESCENT, gst.STEEP_DESCENT]:
                segment.feature_type = tft.TECHNICAL_DESCENT
    
    def _merge_similar_segments(self, segments: List[Segment]) -> List[Segment]:
        """Merge adjacent segments with matching properties"""
        if not segments:
            return []
        
        merged = [segments[0]]
        for current in segments[1:]:
            last = merged[-1]
            
            if (current.feature_type == last.feature_type and
                current.gradient_type == last.gradient_type):
                
                max_len = (
                    current.feature_type.get_config(self.spot_system)['max_length']
                    if current.feature_type
                    else self.spot_system.min_segment_length * 10
                )
                
                if current.length() + last.length() <= max_len:
                    last.points.extend(current.points)
                    last.end_index = current.end_index
                    continue
                    
            merged.append(current)
        
        return merged