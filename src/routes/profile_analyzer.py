from typing import Any, List, Optional, Dict, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from route_helpers import verify_uniform_sampling
from src.routes.baseline import Baseline
from src.routes.spot import RatingSystem
from src.routes.route_processor import ProcessedRoute
from src.routes.route import GeoPoint
from src.routes.trail_features import GradientSegmentType, ShortFeature, TrailFeatureType
from src.routes.trail_features import TrailFeatureType as tft
from src.routes.trail_features import GradientSegmentType as gst
from scipy.signal import argrelextrema

class ProfilePoint(GeoPoint):

    gradient : float
    baseline : float
    baseline_gradient : float

    """Enhanced geographic point with gradient and baseline data"""
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
        
    @staticmethod
    def distance_between(a : 'ProfilePoint' , b : 'ProfilePoint'):
        dist = b.distance_from_origin - a.distance_from_origin
        return abs(dist)

class ProfileSegment:
    """A continuous trail segment referencing points in the parent Profile"""
    def __init__(self, 
                 start_idx: int,
                 end_idx: int,
                 gradient_type: GradientSegmentType,
                 feature_type: Optional[TrailFeatureType] = None):
        self.start_index = start_idx  # Index in parent Profile's points list
        self.end_index = end_idx      # Index in parent Profile's points list
        self.gradient_type = gradient_type
        self.feature_type = feature_type
        self.short_features = []
        self._technical_score = None

    def get_points(self, profile_points: List[ProfilePoint]) -> List[ProfilePoint]:
        """Get the points belonging to this segment from the parent profile"""
        return profile_points[self.start_index:self.end_index+1]
    
    def length(self, profile_points: List[ProfilePoint]) -> float:
        """Get segment length in meters using parent profile points"""
        points = self.get_points(profile_points)
        return sum([ ProfilePoint.distance_between(points[i], points[i - 1]) for i in range(1,len(profile_points)) ])

    def avg_gradient(self, profile_points: List[ProfilePoint]) -> float:
        """Calculate average gradient using parent profile points"""
        points = self.get_points(profile_points)
        return np.mean([p.baseline_gradient for p in points])

    def max_gradient(self, profile_points: List[ProfilePoint]) -> float:
        """Get maximum gradient (positive for ascents, negative for descents)"""
        points = self.get_points(profile_points)
        gradients = [p.gradient for p in points]
        return max(gradients) if self.gradient_type in [
            gst.ASCENT, gst.STEEP_ASCENT, gst.FLAT
        ] else min(gradients)

    def calculate_technical_score(self, spot_system: RatingSystem, profile_points: List[ProfilePoint]) -> float:
        """Calculate comprehensive technical difficulty score"""
        if self._technical_score is not None:
            return self._technical_score
            
        points = self.get_points(profile_points)
        seg_length = self.length(profile_points)
        base_score = 0
        
        # Main feature contribution
        if self.feature_type:
            config = self.feature_type.get_config(spot_system)
            base_score += config.get('difficulty_impact', 0) * (seg_length/100)
        
        # Short features contribution
        feature_score = sum(
            sf.feature_type.get_config(spot_system).get('difficulty_impact', 0) * (sf.length/10)
            for sf in self.short_features
        )
        
        # Gradient modifier
        grad_mod = abs(self.avg_gradient(profile_points)) * 10
        if self.gradient_type in [gst.STEEP_ASCENT, gst.STEEP_DESCENT]:
            grad_mod *= 1.5
        
        self._technical_score = base_score + feature_score + grad_mod
        return self._technical_score

    def validate(self, spot_system: RatingSystem, profile_points: List[ProfilePoint]) -> bool:
        """Validate segment against system requirements"""
        min_length = (spot_system.min_steep_length 
                    if self.gradient_type in (gst.STEEP_ASCENT, gst.STEEP_DESCENT)
                    else spot_system.min_segment_length)
        
        # Feature-specific validation
        if self.feature_type:
            config = self.feature_type.get_config(spot_system)
            if not (config['min_length'] <= self.length(profile_points) <= config['max_length']):
                return False
        
        # Validate short features
        for sf in self.short_features:
            if not sf.validate(spot_system):
                return False
        
        return self.length(profile_points) >= min_length

    def gradient_oscillates(self, profile_points: List[ProfilePoint]) -> bool:
        """Check if segment qualifies for roller analysis"""
        if self.gradient_type in (gst.STEEP_DESCENT, gst.STEEP_ASCENT):
            return False
        
        points = self.get_points(profile_points)
        gradients = [p.gradient for p in points]
        gradient_changes = np.diff(np.sign(gradients))
        reversal_count = np.sum(np.abs(gradient_changes) > 0)
        return reversal_count > 2

    def get_feature_metrics(self, profile_points: List[ProfilePoint]) -> Dict[str, Any]:
        """Get comprehensive feature metrics"""
        points = self.get_points(profile_points)
        metrics = {
            'length': self.length(profile_points),
            'avg_gradient': self.avg_gradient(profile_points),
            'max_gradient': self.max_gradient(profile_points),
            'elevation_change': points[-1].elevation - points[0].elevation,
            'feature_type': self.feature_type.name if self.feature_type else None,
            'feature_count': len(self.short_features)
        }
        
        if self.feature_type in [tft.ROLLER, tft.FLOW_DESCENT]:
            metrics['wavelength'] = self._calculate_wavelength(profile_points)
        
        return metrics

    def _calculate_wavelength(self, profile_points: List[ProfilePoint]) -> Optional[float]:
        """Calculate dominant wavelength for roller/flow segments"""
        points = self.get_points(profile_points)
        residuals = np.array([p.elevation - p.baseline for p in points])
        distances = [p.distance_from_origin for p in points]
        
        peaks = argrelextrema(residuals, np.greater, order=2)[0]
        if len(peaks) < 2:
            return None
            
        wavelengths = [distances[peaks[i]] - distances[peaks[i-1]] 
                      for i in range(1, len(peaks))]
        return float(np.median(wavelengths)) if wavelengths else None

    def create_subsegment(self, start_idx: int, end_idx: int) -> 'ProfileSegment':
        """Create new segment from subset of points within this segment"""
        if start_idx < self.start_index or end_idx > self.end_index:
            raise ValueError("Subsegment indices must be within parent segment")
            
        return ProfileSegment(
            start_idx=start_idx,
            end_idx=end_idx,
            gradient_type=self.gradient_type,
            feature_type=self.feature_type
        )

    def truncate_to_feature(self, spot_system: RatingSystem, profile_points: List[ProfilePoint]) -> Optional['ProfileSegment']:
        """Create a new segment truncated to just the feature portion"""
        if not self.feature_type:
            return None
            
        config = self.feature_type.get_config(spot_system)
        max_length = config.get('max_length', float('inf'))
        
        if self.length(profile_points) <= max_length:
            return self
            
        # Calculate how much to truncate from each end
        points = self.get_points(profile_points)
        target_length = min(max_length, self.length(profile_points))
        
        # Find the most interesting part (highest gradient variance)
        window_size = int(len(points) * 0.3)  # Look at 30% windows
        max_variance = 0
        best_start = 0
        
        for i in range(len(points) - window_size):
            window_gradients = [p.gradient for p in points[i:i+window_size]]
            current_variance = np.var(window_gradients)
            if current_variance > max_variance:
                max_variance = current_variance
                best_start = i
        
        best_start_idx  = self.start_index + best_start

        distances = [p.distance_to_origin for p in profile_points]

        '''This assumes uniform spacing between points! 
        Make sure that this rule is observed is consistenly within this module!'''
        verify_uniform_sampling(distances)

        uniform_spacing = distances[1] - distances[0]
        best_end_idx = min(
            best_start_idx + int(target_length / uniform_spacing),
            self.end_index
        )
        
        return self.create_subsegment(best_start_idx, best_end_idx)

class Profile:

    points : List[ProfilePoint]
    segments : List[ProfileSegment]

    """Complete trail profile with classification and analysis"""
    def __init__(self, spot_system: RatingSystem, proute: ProcessedRoute):
        self.spot_system = spot_system
        self.points = self._create_profile_points(proute)
        self.segments = self._build_segments(proute.baseline)
           
    def _create_profile_points(self, proute: ProcessedRoute) -> List[ProfilePoint]:
        """Generate profile points from processed route"""
        t_values = np.linspace(0, 1, len(proute.smooth_points))
        gradients = [proute.get_gradient(t) for t in t_values]
        t = lambda i: proute.baseline.x[i]/proute.baseline.x[-1]
        
        return [
            ProfilePoint(
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
    
    def _build_segments(self, baseline : Baseline) -> List[ProfileSegment]:
        """Construct and classify all segments"""
        raw_segments = self._identify_baseline_segments()
        #for s in raw_segments:
            #print(f"Segment {s.start_index} to {s.end_index}, grade {s.avg_gradient(self.points):.2f}, type {s.gradient_type}")
        processed_segments = []                
        
        for seg in raw_segments:
            
            if seg.gradient_oscillates(self.points):
                
                is_roller = self.attempt_classify_roller(baseline, seg)
                
                if not is_roller:
                    self.attempt_classify_technical(baseline, seg)
                
            self._classify_local_features(seg)
            
            # Truncate segment if it's too long for its feature type
            if seg.feature_type:
                truncated = seg.truncate_to_feature(self.spot_system, self.points)
                if truncated:
                    processed_segments.append(truncated)
                    continue
                    
            processed_segments.append(seg)
        
        #return self._merge_similar_segments(processed_segments)
        return processed_segments

    def _identify_baseline_segments(self) -> List[ProfileSegment]:
        """Identify segments using running average of baseline gradient"""
        if not self.points:
            return []
        
        segments = []
        current_start = 0
        current_avg_gradient = self.points[0].baseline_gradient
        
        def meets_segment_requirements(segment_point_count: int, segment_length: float):
            return (segment_point_count >= self.spot_system.min_segment_points and
                    segment_length >= self.spot_system.min_segment_length)
        
        for i in range(1, len(self.points)):
            point = self.points[i]
            
            # Update running average gradient from current_start to current point
            window_points = self.points[current_start:i+1]
            current_avg_gradient  = np.mean([p.baseline_gradient for p in window_points])
            current_verified_type = gst.from_gradient(current_avg_gradient, self.spot_system)
            
            # Get instantaneous gradient type for comparison
            point_gradient_type = gst.from_gradient(point.baseline_gradient, self.spot_system)
            
            # Check if we should create a new segment
            if (current_verified_type != point_gradient_type and 
                not current_verified_type.is_transitional_to(point_gradient_type)):
                
                segment_length = ProfilePoint.distance_between(
                    self.points[current_start], 
                    self.points[i-1]  # Previous point marks end of segment
                )
                segment_points = i - current_start
                
                if meets_segment_requirements(segment_points, segment_length):
                    segments.append(ProfileSegment(
                        start_idx=current_start,
                        end_idx=i-1,
                        gradient_type=current_verified_type
                    ))
                    
                    # Start new segment
                    current_start = i
                    current_avg_gradient = point.baseline_gradient
        
        # Add final segment
        if current_start < len(self.points):
            final_avg_gradient = np.mean([p.baseline_gradient for p in self.points[current_start:]])
            final_verified_type = gst.from_gradient(final_avg_gradient, self.spot_system)
            
            segments.append(ProfileSegment(
                start_idx=current_start,
                end_idx=len(self.points)-1,
                gradient_type=final_verified_type
            ))
        
        return segments
    
    def attempt_classify_technical(self, baseline : Baseline, segment: ProfileSegment) -> bool:
        # Only assign technical features if they meet requirements
        if segment.gradient_type in [gst.DESCENT, gst.STEEP_DESCENT]:
            tech_descent = tft.TECHNICAL_DESCENT
            tech_config = tech_descent.get_config(self.spot_system)
            
            # Validate segment meets technical descent requirements
            if (segment.length(self.points) >= tech_config['min_length'] and
                segment.avg_gradient(self.points) <= tech_config['gradient_range'][1]):                
                segment.feature_type = tech_descent
                return True
        
        elif segment.gradient_type in [gst.ASCENT, gst.STEEP_ASCENT]:
            tech_ascent = tft.TECHNICAL_ASCENT
            tech_config = tech_ascent.get_config(self.spot_system)
            
            # Validate segment meets technical ascent requirements
            if (segment.length(self.points) >= tech_config['min_length'] and
                segment.avg_gradient(self.points) >= tech_config['gradient_range'][0]):
                segment.feature_type = tech_ascent        
                return True
            
        return False

    def attempt_classify_roller(self, baseline: Baseline, segment: ProfileSegment) -> bool:
        """Classify roller segments using wavelength analysis"""
        points = segment.get_points(self.points)
        distances = [p.distance_from_origin for p in points]
        residuals = [p.elevation - p.baseline for p in points]
        
        fft_wavelengths     = self._get_fft_based_wavelengths(baseline.freqs, segment.length(self.points))
        extrema_wavelengths = self._get_extrema_based_wavelengths(residuals, distances)
        
        matched_wavelengths = self._find_matching_wavelengths(
            fft_wavelengths, 
            extrema_wavelengths,
            tolerance=self.spot_system.wavelength_match_tolerance
        )        
        
        return self._classify_by_wavelength_patterns(segment, matched_wavelengths)

    def _get_fft_based_wavelengths(self, freqs, segment_length):
        """Convert FFT frequencies to relevant wavelengths"""
        if freqs is None or len(freqs) == 0:
            return []
        
        wavelengths = 1 / freqs
        # Filter wavelengths that could reasonably appear in this segment
        return [w for w in wavelengths if w <= segment_length * 0.8]

    def _get_extrema_based_wavelengths(self, residuals: List[float], distances: List[float]) -> List[float]:
        """Identify wavelengths from local extrema"""
        peaks = argrelextrema(np.array(residuals), np.greater, order=2)[0]
        troughs = argrelextrema(np.array(residuals), np.less, order=2)[0]
        extrema = np.sort(np.concatenate([peaks, troughs]))
        
        if len(extrema) < 2:
            return []
        
        raw_wavelengths = [distances[extrema[i]] - distances[extrema[i-1]] 
                         for i in range(1, len(extrema))]
        
        return self._cluster_wavelengths(raw_wavelengths, eps=self.spot_system.wavelength_clustering_eps)

    def _find_matching_wavelengths(self, fft_wavelengths: List[float], extrema_wavelengths: List[float], tolerance: float) -> List[float]:
        """Match FFT and extrema wavelengths"""
        matches = []
        for fft_w in fft_wavelengths:
            for ext_w in extrema_wavelengths:
                if abs(fft_w - ext_w) <= tolerance * min(fft_w, ext_w):
                    matches.append((fft_w, ext_w))
        
        return list({round((fft+ext)/2, 1) for fft, ext in matches}) if matches else []

    def _classify_by_wavelength_patterns(self, segment: ProfileSegment, wavelengths: List[float]) -> bool:
        
        if not wavelengths:
            return False
        
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
                    return True
                
        return False

    def _cluster_wavelengths(self, raw_wavelengths: List[float], eps: float = 0.5) -> List[float]:
        """Cluster similar wavelengths"""
        if not raw_wavelengths:
            return []
        
        wavelengths = np.array(raw_wavelengths).reshape(-1, 1)
        db = DBSCAN(eps=eps, min_samples=2).fit(wavelengths)
        
        clustered = []
        for label in set(db.labels_):
            if label != -1:
                cluster_points = wavelengths[db.labels_ == label]
                clustered.append(float(np.median(cluster_points)))
        
        return sorted(clustered, reverse=True) if clustered else [float(np.median(wavelengths))]

    def _classify_local_features(self, segment: ProfileSegment):
        """Identify and classify local features"""
        if segment.feature_type:
            return
            
        points = segment.get_points(self.points)
        gradients = np.array([p.gradient for p in points])
        distances = np.array([p.distance_from_origin for p in points])
        avg_gradient = segment.avg_gradient(self.points)
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
    
    def _merge_similar_segments(self, segments: List[ProfileSegment]) -> List[ProfileSegment]:
        """Merge adjacent compatible segments"""
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
                
                combined_length = ProfilePoint.distance_between(self.points[current.end_index],                                                                 
                                                                self.points[last.start_index])
                
                if combined_length <= max_len:
                    # Update the end index of the last segment
                    last.end_index = current.end_index
                    last.short_features.extend(current.short_features)
                    last._technical_score = None  # Invalidate cached score
                    continue
                    
            merged.append(current)
        
        return merged
    
    def calculate_total_technical_score(self) -> float:
        """Calculate total technical difficulty score for entire profile"""
        return sum(seg.calculate_technical_score(self.spot_system, self.points) for seg in self.segments)
    
    def get_feature_density(self, feature_type: TrailFeatureType) -> float:
        """Calculate density of specific feature type (meters per km)"""
        total_feature_length = sum(
            seg.length(self.points) for seg in self.segments 
            if seg.feature_type == feature_type
        )
        total_length = self.points[-1].distance_from_origin
        return (total_feature_length / total_length) * 1000 if total_length > 0 else 0    
    
    def find_steepest_section(self, min_length=50) -> Optional[ProfileSegment]:
        """Find the steepest continuous section meeting length requirements"""
        candidates = []
        for seg in self.segments:
            if seg.length(self.points) >= min_length:
                steepness = abs(seg.avg_gradient(self.points))
                candidates.append((steepness, seg))
        
        return max(candidates, key=lambda x: x[0])[1] if candidates else None