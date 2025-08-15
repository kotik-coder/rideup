from dataclasses import dataclass
from typing import Any, List, Optional, Dict, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from src.routes.rating_system import GradientSegmentType as gst
from src.routes.rating_system import TrailFeatureType as tft
from src.routes.baseline import Baseline
from src.routes.spot import RatingSystem
from src.routes.route_processor import ProcessedRoute
from src.routes.route import GeoPoint
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
    def distance_between(points : List['ProfilePoint'], start, end):
        dist = [ points[i].distance_from_origin - points[i-1].distance_from_origin 
                for i in range(start + 1,end + 1)]
        return np.sum(dist)

class ProfileSegment:

    """A continuous trail segment referencing points in the parent Profile"""
    def __init__(self, 
                 start_idx: int,
                 end_idx: int,
                 gradient_type: gst,
                 feature: Optional['Feature'] = None):
        self.start_index = start_idx  # Index in parent Profile's points list
        self.end_index = end_idx      # Index in parent Profile's points list
        self.gradient_type = gradient_type
        self.feature = feature
        self.short_features = []
        self._technical_score = None

    def get_points(self, profile_points: List[ProfilePoint]) -> List[ProfilePoint]:
        """Get the points belonging to this segment from the parent profile"""
        return profile_points[self.start_index:self.end_index+1]
    
    def length(self, profile_points: List[ProfilePoint], start_index = 0, end_index = -1) -> float:
        """Get segment length in meters using parent profile points"""
        points = self.get_points(profile_points)
        
        if end_index < 0:
            end_index = len(points) - 1
            
        return ProfilePoint.distance_between(points, start_index, end_index)

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
        if self.feature:
            config = self.feature.feature_type.get_config(spot_system)
            base_score += config.get('difficulty_impact', 0) * (seg_length/100)
        
        # Short features contribution
        feature_score = sum(
            sf.feature_type.get_config(spot_system).get('difficulty_impact', 0) * (sf.length(self, profile_points)/10)
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
        if self.feature:
            config = self.feature.feature_type.get_config(spot_system)
            
            if not (config['min_length'] <= self.length(profile_points) <= config['max_length']):
                return False
            
            if 'required_short_features' in config and len(self.short_features) < config['required_short_features']:
                return False
        
        # Validate short features
        for sf in self.short_features:
            if not sf.validate(self, profile_points, spot_system):
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
            'feature_type': self.feature.feature_type.name if self.feature else None,
            'feature_count': len(self.short_features)
        }
        
        if self.feature.feature_type in [tft.ROLLER, tft.FLOW_DESCENT]:
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

@dataclass
class Feature:
    """Enhanced localized feature with physics-based validation"""
    feature_type: tft
    seg_start_index: int  # Relative to segment start
    seg_end_index: int    # Relative to segment start         
    len : float
    grade : float

    def get_absolute_indices(self, segment_start: int) -> Tuple[int, int]:
        """Convert to absolute indices in the profile"""
        return (segment_start + self.seg_start_index, 
                segment_start + self.seg_end_index)                
        
    @staticmethod
    def create_candidate_feature(segment : ProfileSegment, points : List[ProfilePoint], feature_type : tft) -> 'Feature': 
        return Feature(feature_type,
                        0, 
                        segment.end_index - segment.start_index,
                        len   = segment.length(points),
                        grade = segment.avg_gradient(points)
                       )
        
    def length(self, segment : ProfileSegment, points : List[ProfilePoint]):
        len = segment.length(points, self.seg_start_index, self.seg_end_index)
        return len

    def validate(self, segment : ProfileSegment, 
                       points : List[ProfilePoint],
                       rating_system: RatingSystem) -> bool:            
        config = self.feature_type.get_config(rating_system)
        
        length = self.length(segment, points)
        
        # Length validation
        valid_length = (config['min_length'] <= length <= config['max_length'])
        
        # Gradient validation
        min_grad, max_grad = config['gradient_range']
        self.grade = segment.avg_gradient(points)
        valid_gradient = (min_grad <= self.grade <= max_grad)
        
        # Compatibility check
        valid_combo = self.feature_type.is_compatible_with(segment.gradient_type, rating_system)
        
        self._validated = valid_length and valid_gradient and valid_combo
        return self._validated

    def get_difficulty_score(self, segment : ProfileSegment, 
                                   points: List[ProfilePoint], 
                                   rating_system: RatingSystem) -> float:
        
        """Calculate dynamic difficulty score"""
        if not self.validate(segment, points, rating_system):
            return 0.0
            
        config = self.feature_type.get_config(rating_system)
        base_score = config.get('difficulty_impact', 1.0)
        
        # Gradient modifier
        grad_mod = 1 + abs(segment.avg_gradient(points) - config['gradient_range'][0]) * 2
        
        # Length modifier
        length = segment.length(points)
        length_mod = min(2.0, length / config['min_length'])
        
        return base_score * grad_mod * length_mod

class Profile:

    points : List[ProfilePoint]
    segments : List[ProfileSegment]
    spot_system : RatingSystem

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
        
    def _identify_features(self, baseline : Baseline, seg : ProfileSegment):
        #focus on oscillating gradients first
        if seg.gradient_oscillates(self.points):                
            self._attempt_classify_roller(baseline, seg)                
            
        #then find extreme short features, if any
        self._classify_local_features(seg)
        #if needed, adjust technical difficulty for unclassified segments 
        self._attempt_classify_technical(seg, self.spot_system)        
    
    def _build_segments(self, baseline : Baseline) -> List[ProfileSegment]:
        """Construct and classify all segments"""
        raw_segments = self._identify_baseline_segments()
        #for s in raw_segments:
            #print(f"Segment {s.start_index} to {s.end_index}, grade {s.avg_gradient(self.points):.2f}, type {s.gradient_type}")
        processed_segments = []                
        
        for seg in raw_segments:
            
            self._identify_features(baseline, seg)                                
            processed_segments.append(seg)
        
        return raw_segments

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
                    self.points,
                    current_start, 
                    i-1  # Previous point marks end of segment
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
    
    def _attempt_classify_technical(self, segment: ProfileSegment, rating_system : RatingSystem) -> bool:        
        
        gtype = segment.gradient_type

        #ascent
        if gtype == gst.STEEP_ASCENT:            
            candidate_type = tft.TECHNICAL_ASCENT     
        #descent    
        elif gtype == gst.STEEP_DESCENT:            
            candidate_type = tft.TECHNICAL_DESCENT
        #none
        else:
            return False
        
        candidate = Feature.create_candidate_feature(segment, self.points, candidate_type)
        
        #candidate exists -> check if it is a valid assumption
        if not candidate.validate(segment, self.points, rating_system):                    
            return False
                
        #assign to dominant feature
        segment.feature = candidate
        return True

    def _attempt_classify_roller(self, baseline: Baseline, segment: ProfileSegment) -> bool:
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
                    segment.feature = Feature.create_candidate_feature(segment, self.points, feature_type)
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
        if segment.short_features and len(segment.short_features) > 0:
            segment.short_features.clear()
            
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
            
            existing = any(
                sf.seg_start_index == start_idx and
                sf.seg_end_index   == end_idx
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
                Feature(
                    ftr_type,
                    seg_start_index=start_idx,
                    seg_end_index  =end_idx,
                    len   = segment.length(self.points, start_idx, end_idx),
                    grade = segment.max_gradient(self.points)
                )
            )             
    
    def calculate_total_technical_score(self) -> float:
        """Calculate total technical difficulty score for entire profile"""
        return sum(seg.calculate_technical_score(self.spot_system, self.points) for seg in self.segments)
    
    def get_feature_density(self, feature_type: tft) -> float:
        """Calculate density of specific feature type (meters per km)"""
        total_feature_length = sum(
            seg.length(self.points) for seg in self.segments 
            if seg.feature.feature_type == feature_type
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