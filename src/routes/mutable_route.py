# mutable_route.py
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum
import numpy as np
import networkx as nx
from scipy import stats
from rtree import index

from src.routes.route import GeoPoint, Route

class SegmentType(Enum):
    COMMON = "common"
    FORK = "fork"
    DIVERSION = "diversion"
    ALTERNATIVE = "alternative"
    MERGE = "merge"
    CROSSING = "crossing"

@dataclass
class RouteSegment:
    """Represents a continuous segment of a route with metadata"""
    points: List[GeoPoint]
    source_route_names: Set[str] = field(default_factory=set)
    confidence: float = 1.0
    segment_type: SegmentType = SegmentType.COMMON
    start_junction: Optional[GeoPoint] = None
    end_junction: Optional[GeoPoint] = None
    
    def length(self) -> float:
        """Calculate approximate segment length in meters"""
        if len(self.points) < 2:
            return 0.0
        return sum(self.points[i].distance_to(self.points[i+1]) 
                  for i in range(len(self.points)-1))
    
    def add_source_route(self, route_name: str):
        self.source_route_names.add(route_name)
        self.confidence = len(self.source_route_names)

@dataclass
class EstablishedRoute(Route):
    """Container for statistically verified merged routes that inherits from Route"""
    segments:        List[RouteSegment] = field(default_factory=list)
    original_routes: Dict[str, Route]   = field(default_factory=dict)
    junction_points: List[GeoPoint]     = field(default_factory=list)
    route_graph:     nx.DiGraph         = field(default_factory=nx.DiGraph)
    is_linear:       bool               = field(default=True)  # Track linearity status
    
    def __init__(self, name: str, segments: List[RouteSegment], original_routes: Dict[str, Route]):
        # Create unified route from segments
        unified_points = []
        unified_elevations = []
        
        for segment in segments:
            if segment.confidence >= 0.5:  # Use only high-confidence segments
                unified_points.extend(segment.points)
                unified_elevations.extend([p.elevation for p in segment.points])
        
        # Calculate total distance
        total_distance = sum(seg.length() for seg in segments)
        
        # Initialize as a Route
        super().__init__(
            name=name,
            points=unified_points,
            elevations=unified_elevations,
            descriptions=[f"Merged from {len(original_routes)} routes"],
            total_distance=total_distance
        )
        
        # Set EstablishedRoute specific attributes
        self.segments = segments
        self.original_routes = original_routes
        self.junction_points = []
        self.route_graph = nx.DiGraph()
        self.is_linear = True
        
        # Verify linearity of the merged route
        self._verify_linearity()
    
    def _verify_linearity(self):
        """Verify that the merged route forms a continuous, linear path"""
        if len(self.points) < 2:
            self.is_linear = True
            return
            
        # Check 1: Sequential distance consistency
        total_distance = 0
        max_gap = 0
        gap_threshold = 200.0  # meters - maximum allowed gap between consecutive points (was 100.0)
        
        for i in range(len(self.points) - 1):
            distance = self.points[i].distance_to(self.points[i + 1])
            total_distance += distance
            max_gap = max(max_gap, distance)
            
            if distance > gap_threshold:
                self.is_linear = False
                print(f"Warning: EstablishedRoute '{self.name}' has large gap ({distance:.1f}m) between points {i} and {i+1}")
        
        # Check 2: Overall route coherence (start-to-end distance vs cumulative distance)
        if len(self.points) >= 3:
            direct_distance = self.points[0].distance_to(self.points[-1])
            if total_distance > 0 and direct_distance > 0:
                detour_ratio = total_distance / direct_distance
                if detour_ratio > 10.0:  # Was 3.0 — too strict for merged routes with multiple branches
                    self.is_linear = False
                    print(f"Warning: EstablishedRoute '{self.name}' has high detour ratio: {detour_ratio:.2f}")
        
        # Check 3: Bearing consistency (detect backtracking or loops)
        if len(self.points) >= 3:
            bearing_changes = []
            for i in range(1, len(self.points) - 1):
                bearing1 = self.points[i-1].bearing_to(self.points[i])
                bearing2 = self.points[i].bearing_to(self.points[i+1])
                bearing_diff = abs((bearing2 - bearing1 + 180) % 360 - 180)
                bearing_changes.append(bearing_diff)
            
            avg_bearing_change = np.mean(bearing_changes) if bearing_changes else 0
            if avg_bearing_change > 90:  # Average bearing change > 90 degrees suggests non-linearity
                self.is_linear = False
                print(f"Warning: EstablishedRoute '{self.name}' has high average bearing change: {avg_bearing_change:.1f}°")
        
        # Check 4: Point density consistency
        distances = []
        for i in range(len(self.points) - 1):
            distances.append(self.points[i].distance_to(self.points[i + 1]))
        
        if distances:
            avg_distance = np.mean(distances)
            std_distance = np.std(distances)
            if std_distance > avg_distance * 2:  # High variability in point spacing
                self.is_linear = False
                print(f"Warning: EstablishedRoute '{self.name}' has inconsistent point spacing (avg: {avg_distance:.1f}m, std: {std_distance:.1f}m)")
        
        # Check 5: Validate against original routes (if available)
        if self.original_routes and not self.is_linear:
            # Try to improve linearity by reordering points based on original routes
            self._attempt_linearity_correction()
    
    def _attempt_linearity_correction(self):
        """Attempt to correct non-linear routes by reordering points based on original routes"""
        if not self.original_routes or len(self.points) < 3:
            return
            
        # Find the most representative original route
        best_route = None
        best_similarity = 0
        
        for route_name, route in self.original_routes.items():
            if len(route.points) > len(self.points) * 0.5:  # Only consider substantial routes
                similarity = self._calculate_similarity_to_route(route)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_route = route
        
        if best_route and best_similarity > 0.6:
            # Reorder points based on the most similar original route
            self._reorder_points_using_reference(best_route)
            print(f"Info: Reordered points in EstablishedRoute '{self.name}' using reference route '{best_route.name}'")
            
            # Re-verify linearity after correction
            self.is_linear = True
            self._verify_linearity()
    
    def _calculate_similarity_to_route(self, reference_route: Route) -> float:
        """Calculate similarity between this route and a reference route"""
        if not self.points or not reference_route.points:
            return 0.0
            
        # Sample points from both routes
        sample_self = self._sample_points(self.points, 20)
        sample_ref = self._sample_points(reference_route.points, 20)
        
        # Calculate average minimum distance
        total_min_dist = 0
        for p_self in sample_self:
            min_dist = min(p_self.distance_to(p_ref) for p_ref in sample_ref)
            total_min_dist += min_dist
        
        avg_min_dist = total_min_dist / len(sample_self)
        
        # Convert to similarity score (0-1), lower distance = higher similarity
        max_reasonable_dist = 50.0  # meters
        similarity = max(0, 1 - (avg_min_dist / max_reasonable_dist))
        return similarity
    
    def _sample_points(self, points: List[GeoPoint], num_samples: int) -> List[GeoPoint]:
        """Evenly sample points from a list"""
        if len(points) <= num_samples:
            return points
        step = len(points) // num_samples
        return points[::step]
    
    def _reorder_points_using_reference(self, reference_route: Route):
        """Reorder points to match the sequence of a reference route"""
        if len(self.points) < 2 or len(reference_route.points) < 2:
            return
            
        # For each point in this route, find the closest point in reference route
        point_indices = []
        for point in self.points:
            closest_idx = min(range(len(reference_route.points)), 
                             key=lambda i: point.distance_to(reference_route.points[i]))
            point_indices.append(closest_idx)
        
        # Sort points based on reference route order
        sorted_points = [p for _, p in sorted(zip(point_indices, self.points))]
        
        # Update the points if the order changed
        if sorted_points != self.points:
            self.points = sorted_points
            
    def get_unified_route(self, min_confidence: float = 0.5) -> 'EstablishedRoute':
        """Return self since we're already a unified route"""
        return self
    
    def find_alternatives_at_point(self, point: GeoPoint, max_distance: float = 20.0) -> List[RouteSegment]:
        """Find all alternative paths near a given location"""
        alternatives = []
        for segment in self.segments:
            distances = [point.distance_to(p) for p in segment.points]
            if min(distances) <= max_distance:
                alternatives.append(segment)
        return alternatives

@dataclass
class RouteSimilarityConfig:
    max_deviation_distance: float = 12.0   # Realistic for consumer GPS
    min_segment_length: float = 30.0
    similarity_threshold: float = 0.5
    min_matching_points: int = 5           # Allow shorter matches
    fork_angle_threshold: float = 45.0
    search_window_size: int = 20
    cluster_epsilon: float = 8.0           # Allow clustering of nearby points
    statistical_confidence_level: float = 0.95

class RouteSimilarityAnalyzer:
    """Analyzes spatial similarity between routes and identifies common segments"""
    
    def __init__(self, config: Optional[RouteSimilarityConfig] = None):
        self.config = config or RouteSimilarityConfig()
        self.spatial_index = index.Index()
        self.segment_bboxes = {}
    
    def compute_frechet_distance(self, track1: List[GeoPoint], track2: List[GeoPoint]) -> float:
        """Compute discrete Fréchet distance between two point sequences using proper geodesic distance"""
        if not track1 or not track2:
            return float('inf')
        
        m, n = len(track1), len(track2)
        dist_matrix = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                dist_matrix[i, j] = track1[i].distance_to(track2[j])
        
        # Dynamic programming for Fréchet distance
        dp = np.zeros((m, n))
        dp[0, 0] = dist_matrix[0, 0]
        
        for i in range(1, m):
            dp[i, 0] = max(dp[i-1, 0], dist_matrix[i, 0])
        for j in range(1, n):
            dp[0, j] = max(dp[0, j-1], dist_matrix[0, j])
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i, j] = max(min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1]), 
                              dist_matrix[i, j])
        
        return dp[m-1, n-1]
    
    def find_common_segments(self, route1: Route, route2: Route) -> List[Tuple[int, int, int, int]]:
        """Find matching segments between two routes with better filtering"""
        common_segments = []
        points1, points2 = route1.points, route2.points
        
        if len(points1) < 10 or len(points2) < 10:
            return []  # Skip very short routes
        
        # Calculate overall route similarity first
        overall_similarity = self._calculate_route_similarity(route1, route2)
        if overall_similarity < 0.4:  # Only merge routes with at least 40% similarity (was 0.3)
            print(f"DEBUG: Rejecting merge between '{route1.name}' and '{route2.name}' — similarity too low ({overall_similarity:.2f})")
            return []
        
        # Build spatial index for route2
        rtree_idx = index.Index()
        for j, p in enumerate(points2):
            approx_deg_threshold = self.config.max_deviation_distance / 111000
            rtree_idx.insert(j, (p.lon - approx_deg_threshold, p.lat - approx_deg_threshold, 
                                p.lon + approx_deg_threshold, p.lat + approx_deg_threshold))
        
        i = 0
        while i < len(points1) - self.config.min_matching_points:
            best_match = None
            best_length = 0
            
            # Find nearby points in route2
            p1 = points1[i]
            approx_deg_threshold = self.config.max_deviation_distance / 111000
            nearby_indices = list(rtree_idx.intersection(
                (p1.lon - approx_deg_threshold, p1.lat - approx_deg_threshold, 
                 p1.lon + approx_deg_threshold, p1.lat + approx_deg_threshold)
            ))
            
            for j in nearby_indices:
                if self._points_similar(points1[i], points2[j]):
                    length = self._extend_match(points1, points2, i, j)
                    # Require longer matches and better alignment
                    if (length > best_length and 
                        length >= self.config.min_matching_points and
                        self._check_segment_alignment(points1[i:i+length], points2[j:j+length])):
                        best_length = length
                        best_match = (i, i + length - 1, j, j + length - 1)
            
            if best_match:
                start1, end1, start2, end2 = best_match
                print(f"ACCEPTED: Match between '{route1.name}' [{start1}:{end1}] and '{route2.name}' [{start2}:{end2}] (length={best_length})")
                common_segments.append(best_match)
                i = best_match[1] + 5  # Skip ahead to avoid overlapping matches
            else:
                i += 1            
        
        return common_segments
    
    def _calculate_route_similarity(self, route1: Route, route2: Route) -> float:
        """Calculate overall similarity between two routes (0-1)"""
        # Sample points from both routes for comparison
        sample_points1 = self._sample_route_points(route1.points, 50)
        sample_points2 = self._sample_route_points(route2.points, 50)
        
        # Calculate percentage of points that are close
        close_points = 0
        for p1 in sample_points1:
            for p2 in sample_points2:
                if p1.distance_to(p2) <= self.config.max_deviation_distance * 2:  # More lenient for overall similarity
                    close_points += 1
                    break
        
        return close_points / len(sample_points1)
    
    def _sample_route_points(self, points: List[GeoPoint], num_samples: int) -> List[GeoPoint]:
        """Sample points evenly along the route"""
        if len(points) <= num_samples:
            return points
        step = len(points) // num_samples
        return points[::step]
    
    def _check_segment_alignment(self, seg1: List[GeoPoint], seg2: List[GeoPoint]) -> bool:
        """Check if two segments have similar direction and curvature"""
        if len(seg1) < 3 or len(seg2) < 3:
            return True  # Too short to check alignment
        
        # Calculate average bearing difference
        bearing_diffs = []
        for i in range(1, min(len(seg1), len(seg2))):
            bearing1 = seg1[i-1].bearing_to(seg1[i])
            bearing2 = seg2[i-1].bearing_to(seg2[i])
            diff = abs((bearing1 - bearing2 + 180) % 360 - 180)
            bearing_diffs.append(diff)
        
        avg_diff = np.mean(bearing_diffs)
        return avg_diff < 30  # Allow up to 30 degrees average bearing difference (was 45)
    
    def _points_similar(self, p1: GeoPoint, p2: GeoPoint) -> bool:
        """Check if two points are spatially similar using proper geodesic distance"""
        return p1.distance_to(p2) <= self.config.max_deviation_distance
    
    def _extend_match(self, points1: List[GeoPoint], points2: List[GeoPoint], 
                     start1: int, start2: int) -> int:
        """Extend a matching segment as far as possible"""
        length = 1
        max_length = min(len(points1) - start1, len(points2) - start2)
        
        for offset in range(1, max_length):
            idx1, idx2 = start1 + offset, start2 + offset
            if not self._points_similar(points1[idx1], points2[idx2]):
                break
            length += 1
        
        return length

class HierarchicalRouteMerger:
    """Main class for hierarchical route merging with statistical analysis"""
    
    def __init__(self, similarity_analyzer: RouteSimilarityAnalyzer):
        self.analyzer = similarity_analyzer
        self.route_graph = nx.DiGraph()
        self.segment_registry: Dict[str, RouteSegment] = {}
        self.segment_clusters: Dict[int, List[RouteSegment]] = {}
    
    def build_route_graph(self, routes: List[Route]) -> None:
        """Build a graph representation with better filtering"""
        # First, filter out routes that are too different
        similar_route_pairs = []
        for i, route1 in enumerate(routes):
            for j, route2 in enumerate(routes[i+1:], i+1):
                # Check basic compatibility first
                if self._are_routes_compatible(route1, route2):
                    print(f"DEBUG: Considering merge between '{route1.name}' and '{route2.name}'")
                    similar_route_pairs.append((route1, route2))
                else:
                    print(f"DEBUG: Rejecting merge between '{route1.name}' and '{route2.name}' — not compatible")
        
        # Then process only compatible pairs
        all_segments = []
        for route1, route2 in similar_route_pairs:
            route_name1 = route1.name
            route_name2 = route2.name
            
            self.route_graph.add_node(route_name1, route=route1, type='route')
            self.route_graph.add_node(route_name2, route=route2, type='route')
            
            common_segments = self.analyzer.find_common_segments(route1, route2)
            
            for seg_idx, (start1, end1, start2, end2) in enumerate(common_segments):
                segment_points1 = route1.points[start1:end1+1]
                segment_points2 = route2.points[start2:end2+1]
                
                # Only create segments for substantial matches
                if len(segment_points1) >= self.analyzer.config.min_matching_points:
                    segment = RouteSegment(
                        points=segment_points1,
                        source_route_names={route_name1, route_name2},
                        confidence=2.0
                    )
                    all_segments.append(segment)
        
        # Only cluster if we have substantial segments
        if all_segments:
            clustered_segments = self._cluster_segments(all_segments)
            
            for cluster_id, segments in clustered_segments.items():
                # Only create merged segments for clusters with multiple source routes
                source_routes = set()
                for seg in segments:
                    source_routes.update(seg.source_route_names)
                
                if len(source_routes) > 1:  # Only merge if multiple routes contribute
                    merged_segment = self._merge_segment_cluster(segments)
                    
                    segment_id = f"cluster_{cluster_id}"
                    self.segment_registry[segment_id] = merged_segment
                    self.route_graph.add_node(segment_id, segment=merged_segment, type='segment')
                        
                    for route_name in merged_segment.source_route_names:
                        self.route_graph.add_edge(route_name, segment_id)
                        self.route_graph.add_edge(segment_id, route_name)
    
    def _are_routes_compatible(self, route1: Route, route2: Route) -> bool:
        """Check if two routes are potentially compatible for merging"""
        # Check length ratio - routes should be within 2x length of each other
        len1 = sum(route1.points[i].distance_to(route1.points[i+1]) for i in range(len(route1.points)-1))
        len2 = sum(route2.points[i].distance_to(route2.points[i+1]) for i in range(len(route2.points)-1))
        
        if max(len1, len2) / min(len1, len2) > 2.0:
            return False
        
        # Check bounding box overlap
        bbox1 = self._get_route_bbox(route1)
        bbox2 = self._get_route_bbox(route2)
        
        # Calculate bbox overlap percentage
        overlap_area = self._bbox_overlap(bbox1, bbox2)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        min_area = min(area1, area2)
        if min_area > 0:
            overlap_ratio = overlap_area / min_area
            return overlap_ratio > 0.3  # Require at least 30% overlap
        
        return False
    
    def _get_route_bbox(self, route: Route) -> Tuple[float, float, float, float]:
        """Get bounding box of a route"""
        lats = [p.lat for p in route.points]
        lons = [p.lon for p in route.points]
        return (min(lons), min(lats), max(lons), max(lats))
    
    def _bbox_overlap(self, bbox1: Tuple[float, float, float, float], 
                     bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate area of overlap between two bounding boxes"""
        x_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
        y_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
        return x_overlap * y_overlap
    
    def _cluster_segments(self, segments: List[RouteSegment]) -> Dict[int, List[RouteSegment]]:
        """Cluster segments that overlap spatially using proper distance calculation AND topological constraints"""
        clusters = {}
        
        for i, segment in enumerate(segments):
            if not segment.points:
                continue
            
            centroid = self._calculate_centroid(segment.points)
            clustered = False
            
            # Check if this segment belongs to any existing cluster
            for cluster_id, cluster_segments in clusters.items():
                cluster_centroid = self._calculate_centroid(cluster_segments[0].points)
                
                # Check spatial proximity
                if centroid.distance_to(cluster_centroid) <= self.analyzer.config.cluster_epsilon:
                    # ✅ Check topological connectivity
                    is_connected = False
                    for existing_seg in cluster_segments:
                        if self._segments_are_connected(segment, existing_seg):
                            is_connected = True
                            break
                    
                    if is_connected:
                        clusters[cluster_id].append(segment)
                        clustered = True
                        break
            
            if not clustered:
                clusters[i] = [segment]
        
        return clusters
    
    def _segments_are_connected(self, seg1: RouteSegment, seg2: RouteSegment) -> bool:
        """Check if two segments are topologically connected"""
        if not seg1.points or not seg2.points:
            return False
        
        # Check if they share a junction
        if (seg1.end_junction and seg2.start_junction and
            seg1.end_junction.distance_to(seg2.start_junction) <= self.analyzer.config.max_deviation_distance):
            return True
        if (seg2.end_junction and seg1.start_junction and
            seg2.end_junction.distance_to(seg1.start_junction) <= self.analyzer.config.max_deviation_distance):
            return True
        
        # Check if they overlap spatially (first and last 3 points)
        for p1 in seg1.points[:3] + seg1.points[-3:] if len(seg1.points) > 6 else seg1.points:
            for p2 in seg2.points[:3] + seg2.points[-3:] if len(seg2.points) > 6 else seg2.points:
                if p1.distance_to(p2) <= self.analyzer.config.max_deviation_distance:
                    return True
        
        return False
        
    def _merge_segment_cluster(self, segments: List[RouteSegment]) -> RouteSegment:
        """Merge segments using spatial clustering first, then sequence ordering"""
        if not segments:
            return RouteSegment(points=[], confidence=0.0)
        
        # Collect all points
        all_points = []
        all_source_routes = set()
        
        for segment in segments:
            all_points.extend(segment.points)
            all_source_routes.update(segment.source_route_names)
        
        if not all_points:
            return RouteSegment(points=[], confidence=0.0)
        
        # ✅ STEP 1: Spatial clustering only (ignore sequence/topology for now)
        clustered_points = []
        processed = set()
        
        for i, p in enumerate(all_points):
            if i in processed:
                continue
            cluster = [p]
            processed.add(i)
            for j in range(i + 1, len(all_points)):
                if j not in processed and p.distance_to(all_points[j]) <= self.analyzer.config.cluster_epsilon:
                    cluster.append(all_points[j])
                    processed.add(j)
            clustered_points.append(cluster)
        
        # ✅ STEP 2: Compute representative point for each cluster
        merged_points = []
        for cluster in clustered_points:
            if len(cluster) > 1:
                lats = [pt.lat for pt in cluster]
                lons = [pt.lon for pt in cluster]
                elevs = [pt.elevation for pt in cluster]
                try:
                    lat_mode = lats[np.argmax(stats.gaussian_kde(lats)(lats))]
                    lon_mode = lons[np.argmax(stats.gaussian_kde(lons)(lons))]
                    ele_mean = np.mean(elevs)
                except:
                    lat_mode = np.median(lats)
                    lon_mode = np.median(lons)
                    ele_mean = np.mean(elevs)
                merged_points.append(GeoPoint(lat_mode, lon_mode, ele_mean))
            else:
                merged_points.append(cluster[0])
        
        # ✅ STEP 3: Sort by reference to longest source segment
        if segments:
            longest_segment = max(segments, key=lambda s: len(s.points))
            ref_points = longest_segment.points
            
            def get_sort_key(p):
                return min((p.distance_to(rp), i) for i, rp in enumerate(ref_points))[1]
            
            merged_points.sort(key=get_sort_key)
        
        # ✅ STEP 4: Validate gaps (but allow larger gaps for merged routes)
        for i in range(len(merged_points) - 1):
            gap = merged_points[i].distance_to(merged_points[i + 1])
            if gap > 200.0:  # Allow up to 200m gaps in merged routes (often due to bad ordering)
                print(f"WARNING: Large gap ({gap:.1f}m) in merged segment — may need manual review")
                # But don't reject — let linearity check handle it later
        
        return RouteSegment(
            points=merged_points,
            source_route_names=all_source_routes,
            confidence=len(all_source_routes),
            segment_type=SegmentType.COMMON
        )
    
    def _find_segment_index(self, point_idx: int, segment_boundaries: List[Tuple[int, int]]) -> int:
        """Find which segment a point belongs to"""
        for seg_idx, (start, end) in enumerate(segment_boundaries):
            if start <= point_idx < end:
                return seg_idx
        return 0
    
    def _calculate_centroid(self, points: List[GeoPoint]) -> GeoPoint:
        """Calculate centroid of a set of points"""
        if not points:
            return GeoPoint(0, 0, 0)
        avg_lat = np.mean([p.lat for p in points])
        avg_lon = np.mean([p.lon for p in points])
        avg_ele = np.mean([p.elevation for p in points])
        return GeoPoint(avg_lat, avg_lon, avg_ele)
    
    def merge_routes(self) -> List[EstablishedRoute]:
        """Cluster routes and create EstablishedRoutes from connected components — REJECT NON-LINEAR ROUTES"""
        established_routes = []
        
        for component in nx.weakly_connected_components(self.route_graph):
            route_nodes = [node for node in component if self.route_graph.nodes[node].get('type') == 'route']
            if not route_nodes:
                continue
                
            # Collect segments for this component only
            segments = []
            for node in component:
                node_data = self.route_graph.nodes[node]
                if node_data.get('type') == 'segment' and node in self.segment_registry:
                    segments.append(self.segment_registry[node])
            
            # Cluster segments within this component only
            if segments:
                component_segments = segments[:]  # Copy
                clustered_segments = self._cluster_segments(component_segments)
                
                # Rebuild merged segments
                rebuilt_segments = []
                for cluster_id, segs in clustered_segments.items():
                    merged_seg = self._merge_segment_cluster(segs)
                    if merged_seg.points:  # Only add if not empty
                        rebuilt_segments.append(merged_seg)
                
                segments = rebuilt_segments
            
            # Build topology and create EstablishedRoute
            self._build_topological_graph(segments)
            
            established_route = EstablishedRoute(
                name=f"EstablishedRoute_{len(established_routes)}",
                segments=segments,
                original_routes={node: self.route_graph.nodes[node]['route'] for node in route_nodes}
            )
            
            # ✅ CRITICAL: Only keep routes that pass linearity checks
            if established_route.is_linear:
                established_routes.append(established_route)
                print(f"SUCCESS: Created linear EstablishedRoute '{established_route.name}' from {len(route_nodes)} routes")
            else:
                print(f"REJECTED: Non-linear EstablishedRoute '{established_route.name}' from {len(route_nodes)} routes")
                problematic_routes = list(established_route.original_routes.keys())
                print(f"  Source routes: {problematic_routes}")
        
        return established_routes
    
    def _build_topological_graph(self, segments: List[RouteSegment]):
        """Build topological graph with proper junction detection using geodesic distance"""
        junction_candidates = {}
        
        # First pass: identify all segment endpoints as potential junctions
        for segment in segments:
            if not segment.points:
                continue
                
            start_point, end_point = segment.points[0], segment.points[-1]
            
            # Use distance-based junction matching
            start_found = False
            end_found = False
            
            # Convert junction_candidates to a list of (coords, point) tuples for iteration
            existing_junctions = list(junction_candidates.items())
            
            for existing_coords, existing_junction_data in existing_junctions:
                existing_point = existing_junction_data['point']  # Extract the GeoPoint
                
                if start_point.distance_to(existing_point) <= self.analyzer.config.max_deviation_distance:
                    junction_candidates[existing_coords]['segments'].append((segment, 'start'))
                    start_found = True
                if end_point.distance_to(existing_point) <= self.analyzer.config.max_deviation_distance:
                    junction_candidates[existing_coords]['segments'].append((segment, 'end'))
                    end_found = True
            
            if not start_found:
                junction_candidates[(start_point.lat, start_point.lon)] = {
                    'point': start_point,
                    'segments': [(segment, 'start')]
                }
            if not end_found:
                junction_candidates[(end_point.lat, end_point.lon)] = {
                    'point': end_point,
                    'segments': [(segment, 'end')]
                }
        
        # Second pass: create actual junctions where multiple segments meet
        junctions = {}
        for coords, junction_data in junction_candidates.items():
            junction_segments = junction_data['segments']
            junction_point = junction_data['point']
            
            if len(junction_segments) > 1:
                # This is a real junction
                junctions[coords] = junction_point
                
                # Update segments with junction references
                for segment, endpoint_type in junction_segments:
                    if endpoint_type == 'start':
                        segment.start_junction = junction_point
                    else:
                        segment.end_junction = junction_point
        
        # Third pass: analyze angles to detect forks
        for coords, junction_point in junctions.items():
            junction_data = junction_candidates[coords]
            incoming_segments = []
            outgoing_segments = []
            
            for segment, endpoint_type in junction_data['segments']:
                if endpoint_type == 'end':  # Segment ends at junction
                    incoming_segments.append(segment)
                else:  # Segment starts at junction
                    outgoing_segments.append(segment)
            
            # Analyze angles between incoming and outgoing segments
            self._analyze_junction_angles(junction_point, incoming_segments, outgoing_segments)
    
    def _analyze_junction_angles(self, junction_point: GeoPoint, 
                                incoming_segments: List[RouteSegment],
                                outgoing_segments: List[RouteSegment]):
        """Analyze bearing angles to detect forks and merges using proper geodesic bearings"""
        for incoming in incoming_segments:
            if not incoming.points or len(incoming.points) < 2:
                continue
                
            # Get direction approaching the junction
            approach_point = incoming.points[-2]  # Point before junction
            incoming_bearing = approach_point.bearing_to(junction_point)
            
            for outgoing in outgoing_segments:
                if not outgoing.points or len(outgoing.points) < 2:
                    continue
                    
                # Get direction leaving the junction
                departure_point = outgoing.points[1]  # Point after junction
                outgoing_bearing = junction_point.bearing_to(departure_point)
                
                # Calculate angle difference
                angle_diff = abs((outgoing_bearing - incoming_bearing + 180) % 360 - 180)
                
                if angle_diff > self.analyzer.config.fork_angle_threshold:
                    # This is a fork
                    outgoing.segment_type = SegmentType.FORK

# Utility function for easy integration
def create_route_merger(config: Optional[RouteSimilarityConfig] = None) -> HierarchicalRouteMerger:
    """Convenience function to create a configured route merger — with trail-optimized defaults"""
    analyzer = RouteSimilarityAnalyzer(config)
    return HierarchicalRouteMerger(analyzer)