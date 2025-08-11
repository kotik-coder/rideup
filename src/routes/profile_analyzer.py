from typing import List, Optional

import numpy as np
from sklearn.cluster import DBSCAN
from src.routes.spot import RatingSystem
from src.routes.route_processor import ProcessedRoute
from src.routes.route import GeoPoint
from src.routes.trail_features import ElevationSegment, GradientSegmentType, ShortFeature, TrailFeatureType
from src.routes.trail_features import TrailFeatureType as tft
from src.routes.trail_features import GradientSegmentType as gst

from scipy.signal import argrelextrema

class StaticProfilePoint(GeoPoint):
    
    gradient : float
    baseline : float
    baseline_gradient : float
    feature_type  : TrailFeatureType    
    gradient_type : GradientSegmentType
    
    """Enhanced GeoPoint with gradient and segment data"""
    def __init__(self, 
                 lat: float, 
                 lon: float, 
                 elevation: float, 
                 distance_from_origin: float, 
                 gradient: float,
                 gradient_type: GradientSegmentType = gst.FLAT,
                 feature_type: Optional[TrailFeatureType] = None,
                 baseline: float = 0,
                 baseline_gradient: float = 0):
        super().__init__(lat, lon, elevation, distance_from_origin)
        self.gradient = gradient
        self.gradient_type = gradient_type
        self.feature_type = feature_type
        self.baseline = baseline
        self.baseline_gradient = baseline_gradient

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            'gradient': self.gradient,
            'gradient_type': self.gradient_type.name if self.gradient_type else None,
            'feature_type': self.feature_type.name if self.feature_type else None
        })
        return base
    
class StaticProfile:
    
    points   : List[StaticProfilePoint]
    spot_system : RatingSystem
    
    def __init__(self, spot_system : RatingSystem, proute: ProcessedRoute):
        self.spot_system = spot_system
        t_values = np.linspace(0, 1, len(proute.smooth_points))
        gradients = [proute.get_gradient(t) for t in t_values]
        t = lambda i : proute.baseline.x[i]/proute.baseline.x[-1]
        
        self.points = [ StaticProfilePoint(
                    p.lat,
                    p.lon, 
                    p.elevation,
                    p.distance_from_origin,
                    gradient=gradients[i],
                    gradient_type=gst.from_gradient(gradients[i], spot_system),
                    baseline=proute.baseline.y[i],
                    baseline_gradient=proute.baseline.get_baseline_gradient(t(i))
                )
                for i, p in enumerate(proute.smooth_points)
                ]        

    def classify_roller(self, segment: ElevationSegment, route: ProcessedRoute):
        """Classify roller segments using multi-wavelength analysis"""
        distances = [self.points[i].distance_from_origin 
                    for i in range(segment.start_index, segment.end_index+1)]
        segment_length = distances[-1] - distances[0]
        residual = route.get_oscillations(segment.start_index, segment.end_index)
        
        # Get all candidate wavelengths from both methods
        fft_wavelengths     = self._get_fft_based_wavelengths(route.baseline.freqs, segment_length)
        extrema_wavelengths = self._get_extrema_based_wavelengths(residual, distances)        
        
        # Find matching wavelength patterns
        matched_wavelengths = self._find_matching_wavelengths(
            fft_wavelengths, extrema_wavelengths, tolerance=self.spot_system.wavelength_match_tolerance
        )
        
        # Classification logic based on matched patterns
        if matched_wavelengths:
            self._classify_by_wavelength_patterns(segment, matched_wavelengths)

    def _get_fft_based_wavelengths(self, freqs, segment_length):
        """Convert FFT frequencies to relevant wavelengths"""
        if freqs is None or len(freqs) == 0:
            return []
        
        wavelengths = 1 / freqs
        # Filter wavelengths that could reasonably appear in this segment
        return [w for w in wavelengths if w <= segment_length * 0.8]

    def _get_extrema_based_wavelengths(self, residual, distances):
        """Get wavelengths from local extrema with quality checks"""
        peaks   = argrelextrema(residual, np.greater, order=2)[0]
        troughs = argrelextrema(residual, np.less,    order=2)[0]
        extrema = np.sort(np.concatenate([peaks, troughs]))
        
        if len(extrema) < 2:
            return []
        
        # Calculate distances between consecutive extrema
        raw_wavelengths = [distances[extrema[i]] - distances[extrema[i-1]] 
                        for i in range(1, len(extrema))]
        
        # Cluster similar wavelengths to find dominant local patterns
        return self._cluster_wavelengths(raw_wavelengths, eps=self.spot_system.wavelength_clustering_eps)

    def _find_matching_wavelengths(self, fft_wavelengths, extrema_wavelengths, tolerance):
        """Find wavelengths present in both FFT and extrema analysis"""
        matches = []
        
        for fft_w in fft_wavelengths:
            for ext_w in extrema_wavelengths:
                if abs(fft_w - ext_w) <= tolerance * min(fft_w, ext_w):
                    matches.append((fft_w, ext_w))
        
        # Return unique matched wavelengths (average of matched pairs)
        if not matches:
            return []
        
        # Group similar matches and take representative values
        return list({round((fft+ext)/2, 1) for fft, ext in matches})

    def _classify_by_wavelength_patterns(self, segment, wavelengths):
        if not wavelengths:
            return None
        
        # No matching wavelength in flow range
        if segment.gradient_type in [gst.DESCENT, gst.STEEP_DESCENT]:
            segment.feature_type = tft.TECHNICAL_DESCENT        
            
        elif segment.gradient_type in [gst.ASCENT, gst.STEEP_ASCENT]:
            segment.feature_type = tft.TECHNICAL_ASCENT        
        
        """Determine feature type based on wavelength patterns"""
        for w in sorted(wavelengths):
            if self.spot_system.flow_wavelength_min <= w <= self.spot_system.flow_wavelength_max:
                                
                if segment.gradient_type in [gst.DESCENT, gst.STEEP_DESCENT]:
                    segment.feature_type = tft.FLOW_DESCENT
                else:
                    segment.feature_type = tft.ROLLER                    
                    
                break                
    
    def _cluster_wavelengths(self, raw_wavelengths, eps=0.5):
        """Group similar wavelengths to find dominant patterns using DBSCAN clustering"""
        if not raw_wavelengths:
            return []
        
        # Convert to 2D array for clustering (sklearn expects this format)
        wavelengths = np.array(raw_wavelengths).reshape(-1, 1)
        
        # DBSCAN handles outlier rejection automatically
        # eps: maximum distance between two samples to be considered in the same neighborhood
        # min_samples: minimum number of samples in a neighborhood to form a cluster
        db = DBSCAN(eps=eps, min_samples=2).fit(wavelengths)
        
        cluster_labels = db.labels_
        
        # Get cluster centers (representative wavelengths)
        clustered_wavelengths = []
        for label in set(cluster_labels):
            if label != -1:  # -1 is noise in DBSCAN
                cluster_points = wavelengths[cluster_labels == label]
                clustered_wavelengths.append(float(np.median(cluster_points)))
        
        # Sort by cluster size (most frequent patterns first)
        if clustered_wavelengths:
            clustered_wavelengths.sort(reverse=True)
            return clustered_wavelengths
        
        # If no clusters found, return the median of all wavelengths
        return [float(np.median(wavelengths))]
    
class SegmentProfile:
    
    segments : List[ElevationSegment]
    spot_system : RatingSystem
        
    def __init__(self, profile: StaticProfile, route: ProcessedRoute):
        self.spot_system = profile.spot_system
        # Step 1: Identify raw segments (baseline-dominant)
        self._identify_baseline_trends(profile)
        
        # Step 2: Classify by elevation layer        
        for seg in self.segments:
            # Meso-features (residual-dominant)
            if seg.is_roller_candidate():
                profile.classify_roller(seg, route)
            
            # Analyze actual gradient trends for local features
            self._classify_local_features(seg)
        
        self._post_process_segments()

    def _classify_local_features(self, segment: ElevationSegment):
        """Analyze stored gradient trends to identify local features"""
        gradients = np.array(segment.gradients)
        distances = np.array(segment.distances)
        avg_gradient = segment.avg_gradient()
        gradient_std = np.std(gradients)
        
        # Calculate thresholds for extreme gradients (2 sigma)
        upper_threshold = avg_gradient + (2 * gradient_std)
        lower_threshold = avg_gradient - (2 * gradient_std)
        
        # Identify extreme gradient points
        extreme_points = np.where(
            (gradients > upper_threshold) | (gradients < lower_threshold)
        )[0]
        
        # Group consecutive extreme points into features
        extreme_features = []
        if len(extreme_points) > 0:
            current_start = extreme_points[0]
            for i in range(1, len(extreme_points)):
                if extreme_points[i] != extreme_points[i-1] + 1:
                    # End of current feature
                    extreme_features.append((current_start, extreme_points[i-1]))
                    current_start = extreme_points[i]
            extreme_features.append((current_start, extreme_points[-1]))
        
        # Classify extreme features
        for start_idx, end_idx in extreme_features:
            # Calculate feature length using distances
            feature_length = distances[end_idx] - distances[start_idx]

            # Skip features that are too long or have zero length
            if (feature_length >= self.spot_system.step_feature_max_length or 
                feature_length <= 0):
                continue
                
            feature_gradients = gradients[start_idx:end_idx+1]
            avg_feature_gradient = np.mean(feature_gradients)
            max_feature_gradient = (np.max(feature_gradients) if avg_feature_gradient > 0 
                                else np.min(feature_gradients))
            
            # Check if this feature already exists
            existing = any(
                sf.start_index == segment.start_index + start_idx and
                sf.end_index == segment.start_index + end_idx
                for sf in segment.short_features
            )
            if existing:
                continue
                
            # Classify feature type
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

            # Validate feature compatibility before adding
            if not self.spot_system.is_feature_compatible(ftr_type.name, gradient_type.name):
                continue

            # Validate feature length against RatingSystem parameters
            config = ftr_type.get_config(self.spot_system)
            if not (config['min_length'] <= feature_length <= config['max_length']):
                continue
            
            segment.short_features.append(
                ShortFeature(
                    feature_type=ftr_type,
                    gradient_type=gradient_type,  # Added this required parameter
                    start_index=segment.start_index + start_idx,
                    end_index=segment.start_index + end_idx,
                    max_gradient=max_feature_gradient,
                    length=feature_length
                )
            ) 
        
        if not segment.feature_type:

            # Classify segment as Technical if it has multiple unique short features
            unique_features = set(
                (f.start_index, f.end_index) for f in segment.short_features
            )
            if len(unique_features) >= 2:
                if segment.gradient_type in [gst.ASCENT, gst.STEEP_ASCENT]:
                    segment.feature_type = tft.TECHNICAL_ASCENT
                elif segment.gradient_type in [gst.DESCENT, gst.STEEP_DESCENT]:
                    segment.feature_type = tft.TECHNICAL_DESCENT 

    def _post_process_segments(self):
        """Phase 1 post-processing focused on ride context"""
        
        #for i, seg in enumerate(self.segments):
            # Tag segments with riding context
            #seg.riding_context = seg.determine_riding_context(self.segments, i)

    def _identify_baseline_trends(self, profile: StaticProfile):
        """Identifies continuous segments with consistent gradient characteristics"""        
        self.segments = []
        current_segment = None

        for i, point in enumerate(profile.points):

            # Determine gradient classification
            current_gradient_type = gst.from_gradient(point.baseline_gradient, self.spot_system)
            
            if current_segment is None:
                # Start new segment
                current_segment = ElevationSegment(
                    start_idx     = i,
                    gradient_type = current_gradient_type,
                    gradient      = point.gradient,
                    distance      = point.distance_from_origin
                )
            else:
                if current_segment.should_continue(current_gradient_type):
                    # Extend current segment
                    current_segment.extend(
                        end_idx=i,
                        gradient=point.gradient,
                        distance=point.distance_from_origin
                    )
                else:
                    # Finalize and start new segment
                    if current_segment.validate(self.spot_system):
                        self.segments.append(current_segment)
                        
                    current_segment = None                        
                    
        if current_segment:
            self.segments.append(current_segment)