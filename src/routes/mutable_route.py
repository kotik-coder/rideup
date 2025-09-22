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

@dataclass
class EstablishedRoute(Route):
    """Container for statistically verified merged routes that inherits from Route"""
    segments:        List[RouteSegment] = field(default_factory=list)
    original_routes: Dict[str, Route]   = field(default_factory=dict)
    junction_points: List[GeoPoint]     = field(default_factory=list)
    route_graph:     nx.DiGraph         = field(default_factory=nx.DiGraph)
    is_linear:       bool               = field(default=True)
    
    def __init__(self, name: str, segments: List[RouteSegment], original_routes: Dict[str, Route]):
        # Reconstruct continuous path from segments using proper topological ordering
        unified_points = self._reconstruct_continuous_path(segments)
        
        # Calculate total distance
        total_distance = sum(seg.length() for seg in segments)
        
        # Initialize as a Route
        super().__init__(
            name=name,
            points=unified_points,
            elevations=[p.elevation for p in unified_points],
            descriptions=[f"Merged from {len(original_routes)} routes"],
            total_distance=total_distance
        )
        
        # Set EstablishedRoute specific attributes
        self.segments = segments
        self.original_routes = original_routes
        self.junction_points = []
        self.route_graph = nx.DiGraph()
        self.is_linear = True
        
        # Verify linearity
        self._verify_linearity()
    
    def _reconstruct_continuous_path(self, segments: List[RouteSegment]) -> List[GeoPoint]:
        """Fixed path reconstruction"""
        if not segments:
            return []
        
        if len(segments) == 1:
            return segments[0].points
        
        # Build directed graph with proper connectivity
        segment_graph = nx.DiGraph()
        for i, seg in enumerate(segments):
            segment_graph.add_node(i, segment=seg, length=len(seg.points))
        
        # Add edges between connected segments with proper direction
        for i in range(len(segments)):
            for j in range(len(segments)):
                if i != j and self._segments_are_connected(segments[i], segments[j]):
                    distance = segments[i].points[-1].distance_to(segments[j].points[0])
                    segment_graph.add_edge(i, j, weight=distance)
        
        try:
            # Find the longest path by number of points
            longest_path = []
            max_length = 0
            
            for start_node in segment_graph.nodes():
                for end_node in segment_graph.nodes():
                    if start_node != end_node and nx.has_path(segment_graph, start_node, end_node):
                        path = nx.shortest_path(segment_graph, start_node, end_node, weight='weight')
                        path_length = sum(segment_graph.nodes[n]['length'] for n in path)
                        
                        if path_length > max_length:
                            max_length = path_length
                            longest_path = path
            
            # Reconstruct points
            if longest_path:
                reconstructed = []
                for node_idx in longest_path:
                    reconstructed.extend(segments[node_idx].points)
                return reconstructed
            
        except nx.NetworkXException:
            pass
        
        return self._fallback_path_reconstruction(segments)

    def _segments_are_connected(self, seg1: RouteSegment, seg2: RouteSegment) -> bool:
        """Check if two segments are connected"""
        if not seg1.points or not seg2.points:
            return False
        
        # Check endpoint proximity
        end_to_start = seg1.points[-1].distance_to(seg2.points[0])
        end_to_end = seg1.points[-1].distance_to(seg2.points[-1])
        start_to_start = seg1.points[0].distance_to(seg2.points[0])
        start_to_end = seg1.points[0].distance_to(seg2.points[-1])
        
        return min(end_to_start, end_to_end, start_to_start, start_to_end) < 50.0

    def _fallback_path_reconstruction(self, segments: List[RouteSegment]) -> List[GeoPoint]:
        """Fallback method for path reconstruction"""
        # Sort segments by confidence and concatenate
        sorted_segments = sorted(segments, key=lambda s: s.confidence, reverse=True)
        unified_points = []
        for segment in sorted_segments:
            if segment.confidence >= 0.5:
                unified_points.extend(segment.points)
        return unified_points

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

@dataclass
class RouteSimilarityConfig:
    min_segment_length: float = 30.0
    similarity_threshold: float = 0.5
    min_matching_points: int = 5
    fork_angle_threshold: float = 45.0
    search_window_size: int = 20
    cluster_epsilon: float = 8.0
    statistical_confidence_level: float = 0.95
    eps = 10.0             # meters — points within 10m are considered "matching"
    lcss_window_size = 50
        

class RouteSimilarityAnalyzer:
    """Analyzes spatial similarity between routes and identifies common segments"""
    
    def __init__(self, config: Optional[RouteSimilarityConfig] = None):
        self.config = config or RouteSimilarityConfig()
        
    def _compute_lcss_core(
        self,
        track1: List[GeoPoint],
        track2: List[GeoPoint],
        eps: float = 10,
        window: int = 50,
        return_alignment: bool = False
    ) -> Tuple[float, Optional[List[Tuple[int, int]]]]:
        """
        Unified LCSS computation.
        Returns:
            - similarity score (float)
            - alignment (List[Tuple[int, int]]) if return_alignment=True, else None
        """
        m, n = len(track1), len(track2)
        if m == 0 or n == 0:
            return 0.0, [] if return_alignment else None

        # DP table for lengths
        dp = np.zeros((m + 1, n + 1), dtype=int)
        backtrack = None

        if return_alignment:
            # 0 = match (diag), 1 = skip track1 (up), 2 = skip track2 (left)
            backtrack = np.zeros((m + 1, n + 1), dtype=int)

        # Fill DP table
        for i in range(1, m + 1):
            j_start = max(1, i - window)
            j_end = min(n + 1, i + window + 1)

            for j in range(j_start, j_end):
                dist = track1[i-1].distance_to(track2[j-1])
                if dist <= eps:
                    dp[i][j] = dp[i-1][j-1] + 1
                    if return_alignment:
                        backtrack[i][j] = 0
                else:
                    if dp[i-1][j] >= dp[i][j-1]:
                        dp[i][j] = dp[i-1][j]
                        if return_alignment:
                            backtrack[i][j] = 1
                    else:
                        dp[i][j] = dp[i][j-1]
                        if return_alignment:
                            backtrack[i][j] = 2

        # Compute similarity
        lcss_length = dp[m][n]
        similarity = lcss_length / min(m, n)  # or max(m, n) — your preference

        # Backtrack for alignment if requested
        alignment = None
        if return_alignment and backtrack is not None:
            alignment = []
            i, j = m, n
            while i > 0 and j > 0:
                if backtrack[i][j] == 0:
                    alignment.append((i-1, j-1))
                    i -= 1
                    j -= 1
                elif backtrack[i][j] == 1:
                    i -= 1
                else:
                    j -= 1
            alignment = alignment[::-1]

        return float(similarity), alignment
    
    def find_common_segments(self, route1: Route, route2: Route) -> List[Tuple[int, int, int, int]]:
        """
        Find contiguous segments where LCSS matched points are close in space.
        Returns list of segments: (start1, end1, start2, end2)
        """
        points1, points2 = route1.points, route2.points

        if len(points1) < 5 or len(points2) < 5:
            return []

        # Auto-tune eps if not set
        if not hasattr(self.config, 'eps') or self.config.eps <= 0:
            self.config.eps = self.calculate_eps_for_lcss(points1, points2)

        # Get LCSS alignment (list of matched index pairs)
        alignment = self.compute_lcss_alignment(points1, points2)

        if not alignment:
            return []

        common_segments = []
        current_segment = None
        min_segment_length = self.config.min_matching_points

        for idx1, idx2 in alignment:
            # Since LCSS only matches points within `eps`, we don't need to re-check distance
            # But you can optionally double-check:
            # distance = points1[idx1].distance_to(points2[idx2])
            # if distance > self.config.eps: continue  # shouldn't happen

            if current_segment is None:
                current_segment = [idx1, idx1, idx2, idx2]
            else:
                # Check if this is a continuation (consecutive in both tracks)
                # Optional: allow small gaps? You can relax this.
                if (idx1 == current_segment[1] + 1) and (idx2 == current_segment[3] + 1):
                    current_segment[1] = idx1
                    current_segment[3] = idx2
                else:
                    # End current segment if long enough
                    if current_segment[1] - current_segment[0] + 1 >= min_segment_length:
                        common_segments.append(tuple(current_segment))
                    # Start new segment
                    current_segment = [idx1, idx1, idx2, idx2]

        # Add final segment
        if current_segment and (current_segment[1] - current_segment[0] + 1 >= min_segment_length):
            common_segments.append(tuple(current_segment))

        return common_segments
    
    def compute_lcss_alignment(self, track1: List[GeoPoint], track2: List[GeoPoint]) -> List[Tuple[int, int]]:
        _, alignment = self._compute_lcss_core(
            track1, track2,
            eps=self.config.eps,
            window=self.config.lcss_window_size,
            return_alignment=True
        )
        return alignment or []
            
    def _calculate_route_similarity(self, route1: Route, route2: Route) -> float:
        """Calculate similarity based on LCSS — minimal adaptation"""
        if len(route1.points) < 5 or len(route2.points) < 5:
            return 0.0
        
        self.config.eps = self.calculate_eps_for_lcss(route1.points, route2.points)
 
        # Compute LCSS similarity (already normalized to [0,1])
        similarity, _ = self._compute_lcss_core(route1.points, 
                                                route2.points, 
                                                self.config.eps, 
                                                self.config.lcss_window_size, 
                                                return_alignment=False)

        return similarity

    def calculate_eps_for_lcss(
    self,
    track1: List[GeoPoint],
    track2: List[GeoPoint],
    multiplier: float = 3.0
    ) -> float:
        """
        Auto-tune eps based on average sampling distance of both tracks.
        Assumes similar sampling rates. Multiplier typically 1.5 - 3.0.
        """
        def avg_segment_distance(track):
            if len(track) < 2:
                return 10.0  # fallback
            total = sum(
                track[i].distance_to(track[i+1])
                for i in range(len(track) - 1)
            )
            return total / (len(track) - 1)

        d1 = avg_segment_distance(track1)
        d2 = avg_segment_distance(track2)
        avg_sampling_dist = (d1 + d2) / 2.0

        eps = multiplier * avg_sampling_dist
        return max(eps, 5.0)  # never go below 5m for safety 

class HierarchicalRouteMerger:
    """Main class for hierarchical route merging with statistical analysis"""
    
    def __init__(self, similarity_analyzer: RouteSimilarityAnalyzer):
        self.analyzer = similarity_analyzer
        self.route_graph = nx.DiGraph()
        self.segment_registry: Dict[str, RouteSegment] = {}

    def _are_routes_compatible(self, route1: Route, route2: Route) -> bool:
        # Check if routes have any spatial proximity at all
        bbox1 = self._get_route_bbox(route1)
        bbox2 = self._get_route_bbox(route2)
        
        # Check if bounding boxes overlap at all
        if not self._bboxes_overlap(bbox1, bbox2):
            print(f"DEBUG: Routes {route1.name} and {route2.name} rejected - no bbox overlap")
            return False
        
        # Check if routes are close enough to potentially be the same path
        # Calculate the minimum distance between any points in the two routes
        min_distance = float('inf')
        for p1 in route1.points[::10]:  # Sample every 10th point for efficiency
            for p2 in route2.points[::10]:
                dist = p1.distance_to(p2)
                if dist < min_distance:
                    min_distance = dist
                if min_distance < self.analyzer.config.max_deviation_distance * 3:  # Early exit
                    break
            if min_distance < self.analyzer.config.max_deviation_distance * 3:
                break
        
        if min_distance > self.analyzer.config.max_deviation_distance * 5:
            print(f"DEBUG: Routes {route1.name} and {route2.name} rejected - too far apart: {min_distance:.1f}m")
            return False
        
        return True
    
    def _get_route_bbox(self, route: Route) -> Tuple[float, float, float, float]:
        """Get bounding box of a route"""
        lats = [p.lat for p in route.points]
        lons = [p.lon for p in route.points]
        return (min(lons), min(lats), max(lons), max(lats))
        
    def _bboxes_overlap(self, bbox1: Tuple[float, float, float, float], 
                    bbox2: Tuple[float, float, float, float]) -> bool:
        """Check if two bounding boxes overlap at all"""
        return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or 
                    bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])

    def build_route_graph(self, routes: List[Route]) -> None:
        """Fixed with compatibility check"""
        if len(routes) < 2:
            return
        
        # Add compatibility check back
        route_pairs = []
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                print(f"Testing route pair {i} -- {j}")
                if self._are_routes_compatible(routes[i], routes[j]):
                    print("Routes are compatible!")
                    similarity = self.analyzer._calculate_route_similarity(routes[i], routes[j])
                    print(f"Similarity is {similarity:.2f}")
                    if similarity >= self.analyzer.config.similarity_threshold:
                        print("Route pair added!")
                        route_pairs.append((routes[i], routes[j], similarity))
        
        # Sort by similarity (highest first)
        route_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"Processing {len(route_pairs)} compatible route pairs")
        for route1, route2, similarity in route_pairs:
            common_segments = self.analyzer.find_common_segments(route1, route2)
            print(f"Found {len(common_segments)} common segments between {route1.name} and {route2.name} (similarity: {similarity:.2f})")
                    
            for seg_idx, (start1, end1, start2, end2) in enumerate(common_segments):
                # Extract and merge the aligned segments
                seg1_points = route1.points[start1:end1+1]
                seg2_points = route2.points[start2:end2+1]
                
                merged_points = self._merge_aligned_segment(seg1_points, seg2_points)
                
                if len(merged_points) >= self.analyzer.config.min_matching_points:
                    segment = RouteSegment(
                        points=merged_points,
                        source_route_names={route1.name, route2.name},
                        confidence=similarity  # Use actual similarity as confidence
                    )
                    
                    segment_id = f"seg_{route1.name}_{route2.name}_{seg_idx}"
                    self.segment_registry[segment_id] = segment
                    
                    # Add to graph
                    self.route_graph.add_node(route1.name, route=route1, type='route')
                    self.route_graph.add_node(route2.name, route=route2, type='route')
                    self.route_graph.add_node(segment_id, segment=segment, type='segment')
                    
                    self.route_graph.add_edge(route1.name, segment_id)
                    self.route_graph.add_edge(route2.name, segment_id)
                    self.route_graph.add_edge(segment_id, route1.name)
                    self.route_graph.add_edge(segment_id, route2.name)
    
    def _merge_aligned_segment(self, points1: List[GeoPoint], points2: List[GeoPoint]) -> List[GeoPoint]:
        """Merge aligned segments using DTW-guided averaging"""
        if not points1 or not points2:
            return points1 or points2
        
        # Use DTW to find optimal alignment
        alignment = self.analyzer.compute_lcss_alignment(points1, points2)
        
        merged_points = []
        used_indices1 = set()
        used_indices2 = set()
        
        for idx1, idx2 in alignment:
            if idx1 in used_indices1 or idx2 in used_indices2:
                continue
                
            if idx1 < len(points1) and idx2 < len(points2):
                p1, p2 = points1[idx1], points2[idx2]
                if p1.distance_to(p2) <= self.analyzer.config.max_deviation_distance:
                    # Weighted average based on point quality (could use elevation std dev etc.)
                    avg_lat = (p1.lat + p2.lat) / 2
                    avg_lon = (p1.lon + p2.lon) / 2
                    avg_ele = (p1.elevation + p2.elevation) / 2
                    merged_points.append(GeoPoint(avg_lat, avg_lon, avg_ele))
                    used_indices1.add(idx1)
                    used_indices2.add(idx2)
        
        return merged_points

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
            confidence=sum(seg.confidence for seg in segments) / len(segments),  # Average confidence
            segment_type=SegmentType.COMMON
        )
        
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
    """Convenience function to create a configured route merger"""
    analyzer = RouteSimilarityAnalyzer(config)
    return HierarchicalRouteMerger(analyzer)