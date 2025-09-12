from dataclasses import dataclass
from typing import List, Optional, Tuple
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
    
    start_abs_idx : int
    end_abs_idx   : int
    gradient_type : gst
    feature       : Optional['Feature']
    short_features : List['Feature']

    """A continuous trail segment referencing points in the parent Profile"""
    def __init__(self, 
                 start_idx: int,
                 end_idx: int,
                 gradient_type: gst,
                 feature: Optional['Feature'] = None):
        self.start_abs_idx = start_idx  # Index in parent Profile's points list
        self.end_abs_idx   = end_idx      # Index in parent Profile's points list
        self.gradient_type = gradient_type
        self.feature       = feature
        self.short_features = []            
        
    def identify_extreme_deviations(self, profile : 'Profile') -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
        points       = self.get_points(profile.points)
        gradients    = np.array([p.gradient for p in points])
        avg_gradient = np.mean(gradients)
        gradient_std = np.std(gradients)
        
        upper_threshold = avg_gradient + (2 * gradient_std)
        lower_threshold = avg_gradient - (2 * gradient_std)
        
        extreme_points = np.where(
            (gradients > upper_threshold) | (gradients < lower_threshold)
        )[0]
        
        if len(extreme_points) < 2:
            return [], []
        
        allowed_features   = [tft.SWITCHBACK, tft.KICKER, tft.DROP]
        max_allowed_length = (1 + 1E-3)*np.max([profile.spot_system.get_feature_config(f)['max_length'] for f in allowed_features])
        
        def try_add(extreme_features, start_id, end_id):
            feature_length = ProfilePoint.distance_between(profile.points, start_id, end_id)
            if feature_length < max_allowed_length: 
                extreme_features.append((start_id, end_id))
        
        extreme_features = []
        current_start = extreme_points[0]
        for i in range(1, len(extreme_points)):
            if extreme_points[i] != extreme_points[i-1] + 1:                
                try_add(extreme_features, current_start, extreme_points[i-1])
                current_start = extreme_points[i]
        
        try_add(extreme_features, current_start, extreme_points[-1])
        
        extreme_gradients = [gradients[ef[0]:ef[1] + 1] for ef in extreme_features]
        
        return extreme_features, extreme_gradients
        
    def _reclassify_segment(self, profile : 'Profile') -> bool:
        """Re-classify a segment based on its actual gradient characteristics"""
        avg_gradient = self.grade(profile.points)
        
        # Determine new gradient type
        new_gradient_type = gst.from_gradient(avg_gradient, profile.spot_system)
        
        # Only update if the gradient type has changed
        if new_gradient_type != self.gradient_type:
            self.gradient_type = new_gradient_type
            # Clear existing feature since gradient type changed
            self.feature = None
            self.short_features = []
            return True
        
        return False

    def get_points(self, profile_points: List[ProfilePoint]) -> List[ProfilePoint]:
        """Get the points belonging to this segment from the parent profile"""
        return profile_points[self.start_abs_idx:self.end_abs_idx+1]
    
    def length(self, profile_points: List[ProfilePoint], start_rel_index = 0, end_rel_index = -1) -> float:
        """Get segment length in meters using parent profile points"""
        points = self.get_points(profile_points)
        _len = len(points)
        
        if start_rel_index < 0 or start_rel_index > end_rel_index:
            start_rel_index = 0
        
        if end_rel_index < 0 or end_rel_index > _len - 1:
            end_rel_index = _len - 1
            
        return ProfilePoint.distance_between(points, start_rel_index, end_rel_index)

    def grade(self, profile_points: List[ProfilePoint], start_rel_idx : int = 0, end_rel_idx : int = -1) -> float:
        """Calculate average gradient using parent profile points"""
        
        if start_rel_idx < 0 or start_rel_idx > end_rel_idx:
            start_rel_idx = 0
        
        if end_rel_idx < 0 or end_rel_idx > self.end_abs_idx:
            end_rel_idx = self.end_abs_idx
        
        points = self.get_points(profile_points)[start_rel_idx:end_rel_idx+1]
        return np.mean([p.baseline_gradient for p in points ])

    def max_gradient(self, profile_points: List[ProfilePoint]) -> float:
        """Get maximum gradient (positive for ascents, negative for descents)"""
        points = self.get_points(profile_points)
        gradients = [p.gradient for p in points]
        return max(gradients) if self.gradient_type in [
            gst.ASCENT, gst.STEEP_ASCENT, gst.FLAT
        ] else min(gradients)

    def validate(self, spot_system: RatingSystem, profile_points: List[ProfilePoint]) -> bool:
        """Validate segment against system requirements"""
        min_length = (spot_system.min_steep_length 
                    if self.gradient_type in (gst.STEEP_ASCENT, gst.STEEP_DESCENT)
                    else spot_system.min_segment_length)
        
        if self.length(profile_points) < min_length:
            return False        
                
        # Validate short features
        for sf in self.short_features:
            if not sf.validate(self, profile_points, spot_system):
                return False
        
        # Feature-specific validation
        if self.feature and not self.feature.validate(self, profile_points, spot_system):
            return False
                
        return True
 
    def oscillation_clusters(self, profile_points: List[ProfilePoint], system: RatingSystem) -> Optional[List[List[int]]]:
        """Check if segment qualifies for roller analysis with feature length constraints"""
        allowed_features = [tft.ROLLERCOASTER, tft.FLOW_DESCENT]
        intersection = [feature for feature in allowed_features if feature in system.get_compatible_features(self.gradient_type)]
        
        if len(intersection) == 0:
            return None
        
        points = self.get_points(profile_points)
        gradients = [p.gradient for p in points]
        
        max_cluster_length = min([ system.get_feature_config(f).get('max_length') for f in intersection])
        
        # Find all gradient sign change indices
        sign_change_indices = []
        for i in range(1, len(gradients)):
            if np.sign( gradients[i]) != np.sign(gradients[i-1]):
                sign_change_indices.append(i)
                
        if len(sign_change_indices) < system.num_oscillations_threshold:
            return None
        
        # Cluster sign changes based on distance from the first element in cluster
        clusters = []
        current_cluster = [sign_change_indices[0]]
        
        for i in range(1, len(sign_change_indices)):
            current_idx = sign_change_indices[i]
            first_in_cluster_idx = current_cluster[0]
            
            # Calculate distance from current sign change to first element in cluster
            distance_from_first = ProfilePoint.distance_between(points, first_in_cluster_idx, current_idx)
            
            if distance_from_first <= max_cluster_length:
                # Within cluster distance, add to current cluster
                current_cluster.append(current_idx)
            else:
                # Too far from first element, start new cluster
                if len(current_cluster) >= system.num_oscillations_threshold:
                    clusters.append(current_cluster)
                current_cluster = [current_idx]
        
        # Add the last cluster if valid
        if len(current_cluster) >= system.num_oscillations_threshold:
            clusters.append(current_cluster)
            
        return clusters

@dataclass
class Feature:
    """Enhanced localized feature with physics-based validation"""
    feature_type: tft
    seg_start_index: int  # Relative to segment start
    seg_end_index: int    # Relative to segment start         
    len : float
    grade : float
    wavelength : Optional[float] = None

    def get_absolute_indices(self, segment_start: int) -> Tuple[int, int]:
        """Convert to absolute indices in the profile"""
        return (segment_start + self.seg_start_index, 
                segment_start + self.seg_end_index)                
        
    def find_tightest_boundaries(self, 
                                profile: 'Profile',
                                segment: ProfileSegment, 
                                baseline: Baseline,
                                classification_method) -> Tuple[int, int]:
        """
        Find the closest valid start and end indices where the feature is still observed.
        Uses binary search for efficient boundary finding.
        classification_method(profile, test_feature, segment, baseline): A function that returns True if the feature is valid at given indices        
        """
        # Binary search for tightest start index
        def find_max_start(low, high):
            result = low
            while low <= high: 
                mid = (low + high) // 2
                test_feature     = Feature.create_candidate_feature(segment, 
                                                                    profile.points, 
                                                                    self.feature_type, 
                                                                    self.wavelength, 
                                                                    mid, 
                                                                    self.seg_end_index)
                if classification_method(profile, test_feature, segment, baseline):
                    result = mid      # valid, so update result
                    low = mid + 1     # try moving start further right
                else:
                    high = mid - 1    # too far, move left
            return result

        # Binary search for tightest end index
        def find_min_end(low, high):
            result = high
            while low <= high:
                mid = (low + high) // 2
                test_feature     = Feature.create_candidate_feature(segment, 
                                                                    profile.points, 
                                                                    self.feature_type, 
                                                                    self.wavelength, 
                                                                    self.seg_start_index, 
                                                                    mid)
                if classification_method(profile, test_feature, segment, baseline):
                    result = mid      # valid, so update result
                    high = mid - 1    # try moving end further left
                else:
                    low = mid + 1     # too far left, move right
            return result

        # Find tightest start index first
        max_start = find_max_start(self.seg_start_index, self.seg_end_index)
        
        # Then find tightest end index based on the found start
        min_end   = find_min_end(max_start, self.seg_end_index)
        
        return max_start, min_end
        
    @staticmethod
    def create_candidate_feature(segment: ProfileSegment, 
                               points : List[ProfilePoint], 
                               feature_type : tft, 
                               w : Optional[float] = 0.0,
                               start_rel_idx : int = -1, 
                               end_rel_idx : int = -1) -> 'Feature': 
        
        return Feature(feature_type,
                        start_rel_idx if start_rel_idx >= 0 else 0, 
                        end_rel_idx if end_rel_idx > 0 else segment.end_abs_idx - segment.start_abs_idx,
                        len   = segment.length(points, start_rel_idx, end_rel_idx),
                        grade = segment.grade(points,  start_rel_idx, end_rel_idx),
                        wavelength = w
                       )
    
        
    def length(self, segment : ProfileSegment, points : List[ProfilePoint]):
        return segment.length(points, self.seg_start_index, self.seg_end_index)    
        
    def calc_grade(self, profile_points : List[ProfilePoint], segment : ProfileSegment, local_feature : bool = False):
        start = segment.start_abs_idx + self.seg_start_index
        end   = segment.start_abs_idx + self.seg_end_index
        
        if not local_feature:
            result = np.mean([p.baseline_gradient for p in profile_points[start:end] ] )
        else:
            result = np.max([p.gradient for p in profile_points[start:end] ] )
        
        return result
    
    def validate(self, segment : ProfileSegment, 
                       points : List[ProfilePoint],
                       rating_system: RatingSystem,
                       local_feature : bool = False) -> bool:                            
        config = self.feature_type.get_config(rating_system)        
        self.len = self.length(segment, points)
        
        # Length validation
        valid_length = (config['min_length'] <= self.len <= config['max_length'])
        
        if not valid_length:
            return False
        
        # Gradient validation
        min_grad, max_grad = config['gradient_range']
        
        self.grade     = self.calc_grade(points, segment, local_feature)
        
        valid_gradient = (min_grad <= self.grade <= max_grad)
        
        if not valid_gradient:
            return False                
        
        # Compatibility check                
        valid_config = rating_system.is_feature_compatible(self.feature_type, segment.gradient_type)
        
        if not valid_config:
            return False        
        
        valid_short_features = True
        
        if 'required_short_features' in config:
            valid_short_features = len(segment.short_features) >= config['required_short_features']
        
        if not valid_short_features:
            return False
        
        valid_wavelength = True
        
        if 'wavelength_range' in config:
            valid_wavelength = config['wavelength_range'][0] <= self.wavelength <= config['wavelength_range'][1]
            
        return valid_wavelength

class Profile:

    points      : List[ProfilePoint]
    segments    : List[ProfileSegment]
    spot_system : RatingSystem

    """Complete trail profile with classification and analysis"""
    def __init__(self, spot_system: RatingSystem, proute: ProcessedRoute):
        self.spot_system = spot_system
        self.points      = self._create_profile_points(proute)
        
        raw_segments     = self._identify_baseline_segments()
        processed_segments = self._build_segments(raw_segments, proute.baseline)
                           
        # Merge transitional segments with no features        
        prev_count    = 1        
        merged_result = []
        new_feature_segments = 1
        old_feature_segments = 0                
        
        while len(processed_segments) < prev_count or new_feature_segments > old_feature_segments:
            prev_count         = len(processed_segments)                        
            
            merged_result      = self._merge_transitional_segments(processed_segments, proute.baseline)            
            old_feature_segments = len([c.feature for c in merged_result if c.feature])            
            
            candidate_segments = self._build_segments(merged_result, proute.baseline)                        
            new_feature_segments = len([c.feature for c in candidate_segments if c.feature])                        
            
            processed_segments = candidate_segments \
                                    if new_feature_segments > old_feature_segments \
                                    else merged_result            
            
        self.segments = processed_segments
           
    def _create_profile_points(self, proute: ProcessedRoute) -> List[ProfilePoint]:
        """Generate profile points from processed route"""
        t_values  = np.linspace(0, 1, len(proute.smooth_points))
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
        
    @staticmethod
    def is_feature_conserved(p : 'Profile', feature: Feature, seg : ProfileSegment, bl : Baseline) -> bool:
        if not feature:
            return False
        
        ftr_start, ftr_end = feature.get_absolute_indices(seg.start_abs_idx)
        
        test_segment = ProfileSegment(ftr_start, ftr_end, seg.gradient_type)
        test_segment._reclassify_segment(p)
        
        identified_feature = p._identify_feature(bl, test_segment)
        
        if not identified_feature:
            return False
        
        validated_feature = identified_feature.validate(test_segment, p.points, p.spot_system)
        
        if not validated_feature:
            return False                
        
        return (identified_feature.feature_type == feature.feature_type)        
        
    def _identify_feature(self, baseline : Baseline, seg : ProfileSegment) -> Optional[Feature]:
        #focus on oscillating gradients first
        ftr = None
        if seg.oscillation_clusters(self.points, self.spot_system):
            ftr = self._attempt_classify_roller(baseline, seg)                                    
            
        #then find extreme short features, if any
        self._classify_local_features(seg)                
        
        #if needed, adjust technical difficulty for unclassified segments 
        if not ftr: 
            ftr = self._attempt_classify_technical(seg, self.spot_system)                                
        
        return ftr    
    
    def create_segment(self,
                       baseline : Baseline, 
                       start : int, 
                       end : int, 
                       type : gst):        
        segment = ProfileSegment(
                start_idx=start,
                end_idx=end,
                gradient_type=type
            )
        segment._reclassify_segment(self)
        segment.feature = self._identify_feature(baseline, segment)
        return segment
                
    def _split_segment_at_index(self, 
                            segment: ProfileSegment, 
                            baseline: Baseline,
                            split_index: int
                            ) -> Tuple[Optional[ProfileSegment], Optional[ProfileSegment]]:
        """
        Split a segment at a specific relative index (single responsibility: index-based split).
        Returns (left_segment, right_segment) where either may be None.
        """
        if not segment or split_index <= 0:
            return None, segment
        
        segment_length = segment.end_abs_idx - segment.start_abs_idx + 1
        if split_index >= segment_length:
            return segment, None
        
        # Ensure split index is within valid bounds
        split_index = min(max(split_index, 0), segment_length - 1)
        
        left_segment = self.create_segment(
            baseline,
            segment.start_abs_idx,
            segment.start_abs_idx + split_index - 1,
            segment.gradient_type
        )
        
        right_segment = self.create_segment(
            baseline,
            segment.start_abs_idx + split_index,
            segment.end_abs_idx,
            segment.gradient_type
        )
        
        return left_segment, right_segment

    def _split_segment_by_indices(self,
                                segment: ProfileSegment,
                                baseline: Baseline,
                                cut_start: int,
                                cut_end: int) -> List[ProfileSegment]:
        """
        Split a segment based on start and end cut indices (single responsibility: range-based split).
        Returns 1-3 segments depending on cut positions.
        """
        
        if cut_end > (segment.end_abs_idx - segment.start_abs_idx - 1):
            return [segment]
        
        segments = []
        
        # Segment before the cut range
        before_first_cut, after_first_cut = self._split_segment_at_index(segment, baseline, cut_start)
        segments.append(before_first_cut)
        
        # The cut segment itself    
        if after_first_cut:
            cut_segment, after_second_cut = self._split_segment_at_index(
                after_first_cut, baseline, cut_end - cut_start
            )        
            
            if cut_segment:            
                segments.append(cut_segment)
            
            # Segment after the cut range
            if after_second_cut:
                segments.append(after_second_cut)                                    
                    
        return segments

    def _split_segment_by_clusters(self, 
                                segment: ProfileSegment, 
                                baseline: Baseline,
                                clusters: List[List[int]]
                                ) -> List[ProfileSegment]:
        """
        Split a segment based on oscillation clusters (single responsibility: cluster-based split).
        Uses _split_segment_at_index internally for clean separation.
        """
        segments = []
        current_segment = segment
        
        # Sort clusters by starting index
        sorted_clusters = sorted(clusters, key=lambda cluster: cluster[0])
        
        for cluster in sorted_clusters:
            cluster_start, cluster_end = cluster[0], cluster[-1]
            
            # Split before cluster (if there's content before the cluster)
            if cluster_start > 0 and current_segment:
                before_cluster, remaining_segment = self._split_segment_at_index(
                    current_segment, baseline, cluster_start
                )
                if before_cluster:
                    segments.append(before_cluster)
                current_segment = remaining_segment
            
            # Split cluster segment from remainder (if we still have a segment to split)
            if current_segment:
                cluster_length = cluster_end - cluster_start + 1
                cluster_segment, remaining_segment = self._split_segment_at_index(
                    current_segment, baseline, cluster_length
                )
                if cluster_segment:
                    segments.append(cluster_segment)
                current_segment = remaining_segment
            
            # If we've exhausted the segment, break early
            if not current_segment:
                break
        
        # Add any remaining segment after last cluster
        if current_segment:
            segments.append(current_segment)
        
        return segments
            
    def _build_segments(self, raw_segments : List[ProfileSegment], baseline: Baseline) -> List[ProfileSegment]:
        """Construct and classify all segments"""        
        processed_segments = []                
        
        for seg in raw_segments:
            # Check for oscillation clusters first
            clusters = seg.oscillation_clusters(self.points, self.spot_system)
            
            if clusters:
                # Use the unified split method with clusters parameter
                cluster_split_segments = self._split_segment_by_clusters(seg, clusters=clusters, baseline=baseline)                
                
                # Process each resulting segment
                for split_seg in cluster_split_segments:
                    self._process_individual_segment(split_seg, baseline, processed_segments)
                    
            else:
                self._process_individual_segment(seg, baseline, processed_segments)                                                
                
        return processed_segments

    def _merge_transitional_segments(self, segments: List[ProfileSegment], baseline: Baseline) -> List[ProfileSegment]:            
        merged = []
        none_feature_indices = [i for i, seg in enumerate(segments) if seg.feature is None]
        i = 0
        n = len(segments)
                        
        while i < n - 1:
                
            current_segment = segments[i]
            next_segment    = segments[i + 1]
            
            are_features_absent   = i in none_feature_indices and (i + 1) in none_feature_indices
            is_segment_dissimilar = not current_segment.gradient_type.is_transitional_to(next_segment.gradient_type) and \
                                    current_segment.gradient_type != next_segment.gradient_type
            
            if not are_features_absent or is_segment_dissimilar:
                merged.append(current_segment)
                i += 1
            else:
                # Merge the two segments
                merged_start = current_segment.start_abs_idx
                merged_end = next_segment.end_abs_idx
                
                bigger_segment = self.create_segment(
                    baseline, merged_start, merged_end, current_segment.gradient_type
                )
                
                merged.append(bigger_segment)
                i += 2  # Skip the next segment since we merged it
                
        if i < n:  # Only append if we haven't processed the last segment
            merged.append(segments[n - 1])
        
        return merged

    def _process_individual_segment(self, segment, baseline, processed_segments):
        """Process a single segment with appropriate feature identification"""
        
        # Regular feature identification
        feature = self._identify_feature(baseline, segment)
        
        def overlapping_segments(s, segments) -> bool:
            for p in segments:
                if (s.start_abs_idx >= p.start_abs_idx) and \
                   (s.end_abs_idx <= p.end_abs_idx):                   
                        return True
                
            return False
        
        if feature:
            segment.feature = feature
            
            start, end = feature.find_tightest_boundaries(
                self, segment, baseline, Profile.is_feature_conserved
            )
            #Use unified split method with cut indices
            feature_split_segments = self._split_segment_by_indices(
                segment, cut_start=start, cut_end=end, baseline=baseline
            )               
            
            processed_segments.extend(feature_split_segments)
        else:
            if not overlapping_segments(segment, processed_segments):
                processed_segments.append(segment)                        

    def _is_segment_in_cluster(self, segment: ProfileSegment, 
                            original_segment: ProfileSegment, 
                            cluster: List[int]) -> bool:
        """Check if a segment corresponds to a specific cluster"""
        cluster_start_abs = original_segment.start_abs_idx + cluster[0]
        cluster_end_abs = original_segment.start_abs_idx + cluster[-1]
        
        return (segment.start_abs_idx == cluster_start_abs and 
                segment.end_abs_idx == cluster_end_abs)

    def _identify_baseline_segments(self) -> List[ProfileSegment]:
        """Identify segments using running average of baseline gradient"""
        segments = []
        current_start = 0
        current_avg_gradient = self.points[0].baseline_gradient
        
        def meets_segment_requirements(count: int, len: float):
            min_points = self.spot_system.min_segment_points
            min_length = self.spot_system.min_segment_length
            return (count >= min_points and len >= min_length)
        
        for i in range(1, len(self.points)):
            point = self.points[i]
            
            # Update running average gradient from current_start to current point
            window_points = self.points[current_start:i+1]
            current_avg_gradient  = np.mean([p.baseline_gradient for p in window_points])
            current_verified_type = gst.from_gradient(current_avg_gradient, self.spot_system)
            
            # Get instantaneous gradient type for comparison
            point_gradient_type = gst.from_gradient(point.baseline_gradient, self.spot_system)
            
            # Check if we should create a new segment
            if (current_verified_type != point_gradient_type and                    #new point is not of the same type
                not current_verified_type.is_transitional_to(point_gradient_type)): #and does not belong a transitional type
                
                new_segment = ProfileSegment(
                        start_idx=current_start,
                        end_idx=i-1,
                        gradient_type=current_verified_type
                    )
                segment_length = new_segment.length(self.points)
                segment_points = i - current_start
                
                if meets_segment_requirements(segment_points, segment_length):
                    segments.append(new_segment)                    
                    # Start new segment
                    current_start = i
                    current_avg_gradient = point.baseline_gradient
        
        # Add final segment
        if current_start < len(self.points):
            final_avg_gradient  = np.mean([p.baseline_gradient for p in self.points[current_start:]])
            final_verified_type = gst.from_gradient(final_avg_gradient, self.spot_system)
            
            segments.append(ProfileSegment(
                start_idx=current_start,
                end_idx=len(self.points)-1,
                gradient_type=final_verified_type
            ))
        
        return segments
    
    def _attempt_classify_technical(self, segment: ProfileSegment, rating_system : RatingSystem) -> Optional[Feature]:        
        
        gtype = segment.gradient_type

        #ascent
        if gtype == gst.STEEP_ASCENT:            
            candidate_type = tft.TECHNICAL_ASCENT     
        #descent    
        elif gtype == gst.STEEP_DESCENT:            
            candidate_type = tft.TECHNICAL_DESCENT
        #none
        else:
            return None
        
        candidate = Feature.create_candidate_feature(segment, self.points, candidate_type)
        
        #candidate created -> check if it meets basic requirements
        if not candidate.validate(segment, self.points, rating_system):                    
            return None
                
        return candidate

    def _attempt_classify_roller(self, baseline: Baseline, segment: ProfileSegment) -> Optional[Feature]:
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
        peaks   = argrelextrema(np.array(residuals), np.greater, order=2)[0]
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

    def _classify_by_wavelength_patterns(self, segment: ProfileSegment, wavelengths: List[float]) -> Optional[Feature]:
        
        if not wavelengths:
            return None
        
        compatible_features = self.spot_system.get_compatible_features(segment.gradient_type)        
        allowed_features    = [tft.FLOW_DESCENT, tft.ROLLERCOASTER]
        candidates = [ feature for feature in allowed_features if feature in compatible_features ]
        
        for w in sorted(wavelengths):
            for feature in candidates:
                                        
                f = Feature.create_candidate_feature(segment, self.points, feature, w)
                if f.validate(segment, self.points, self.spot_system):
                    return f
                
        return None

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
        segment.short_features.clear() 
                        
        extreme_features, extreme_gradients = segment.identify_extreme_deviations(self)
        allowed_feature_types = [tft.DROP, tft.KICKER]
        
        for i, ef in enumerate(extreme_features):
            feature_gradients    = extreme_gradients[i]
            avg_feature_gradient = np.mean(feature_gradients)
            
            start_idx, end_idx = ef
            
            existing = any(
                sf.seg_start_index == start_idx and
                sf.seg_end_index   == end_idx
                for sf in segment.short_features
            )
            if existing:
                continue
                
            gradient_type       = gst.from_gradient(avg_feature_gradient, self.spot_system)
            compatible_features = [f for f in self.spot_system.get_compatible_features(gradient_type) if f in allowed_feature_types]            
                     
            candidate_features = [ Feature(
                    cf,
                    seg_start_index=start_idx,
                    seg_end_index  =end_idx,
                    len   = segment.length(self.points, start_idx, end_idx),
                    grade = np.max(feature_gradients)
                ) for cf in compatible_features ]
            
            for candidate in candidate_features:
                if candidate.validate(segment, self.points, self.spot_system, local_feature=True):
                    segment.short_features.append(candidate)