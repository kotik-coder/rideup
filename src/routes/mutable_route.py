# mutable_route.py
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median, stdev
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum
import numpy as np
from scipy import stats

from src.routes.route import GeoPoint, Route
from src.ui.map_helpers import print_step

@dataclass
class RouteTransition:
    """Represents a transition segment in the merged route."""
    start_idx: int
    start_point: GeoPoint
    end_idx: int
    end_point: GeoPoint
    outlier_count: int
    consensus_count: int
    max_residual_ratio: float

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
    
    def length(self) -> float:
        """Calculate approximate segment length in meters"""
        if len(self.points) < 2:
            return 0.0
        return sum(self.points[i].distance_to(self.points[i+1]) 
                  for i in range(len(self.points)-1))

@dataclass
class EstablishedRoute:
    """Mutable container for merged routes — behaves like a Route but is not a subclass."""
    name: str
    points: List[GeoPoint]
    elevations: List[float]
    descriptions: List[str]
    total_distance: Optional[float]
    
    """Container for statistically verified merged routes that inherits from Route"""
    segments: List[RouteSegment] = field(default_factory=list)
    original_routes: Dict[str, Route] = field(default_factory=dict)
    is_linear: bool = field(default=True)
    std_devs: List[Dict[str, float]] = field(default_factory=list)  # NEW: per-point stats
    
    @classmethod
    def from_segments_and_stats(cls, name: str, segments: List[RouteSegment], original_routes: Dict[str, Route], std_devs: List[Dict[str, float]]):
        unified_points = cls._simple_path_reconstruction(segments)
        return cls(
            name=name,
            points=unified_points,
            elevations=[p.elevation for p in unified_points],
            descriptions=[f"Merged from {len(original_routes)} routes"],
            total_distance=sum(seg.length() for seg in segments),
            segments=segments,
            original_routes=original_routes,
            is_linear=len(unified_points) > 1,
            std_devs=std_devs  # Store stats
        )
        
    @staticmethod
    def _simple_path_reconstruction(segments: List[RouteSegment]) -> List[GeoPoint]:
        if not segments:
            return []
        unified_points = []
        for segment in segments:  # ← Remove confidence filtering
            unified_points.extend(segment.points)
        return unified_points

    def _verify_linearity(self):
        """Placeholder for linearity verification"""
        # Basic check: if we have points, consider it linear for now
        self.is_linear = len(self.points) > 1

@dataclass
class RouteSimilarityConfig:
    min_segment_length: float = 30.0
    similarity_threshold: float = 0.85
    min_matching_points: int = 5
    fork_angle_threshold: float = 45.0
    cluster_epsilon: float = 8.0
    eps = 10.0  # meters — points within 10m are considered "matching"
    lcss_window_size = 50
    outlier_confidence_level: float = 0.95  # 95% confidence for outlier detection

class RouteSimilarityAnalyzer:
    """Analyzes spatial similarity between routes using LCSS with auto-adjustable distance"""
    
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

        # Fill DP table with band constraint
        for i in range(1, m + 1):
            # Window bounds: only consider j within [i-window, i+window]
            j_low = max(1, i - window)
            j_high = min(n, i + window)  # Note: j_high is inclusive
            
            for j in range(j_low, j_high + 1):  # +1 because range is exclusive
                dist = track1[i-1].distance_to(track2[j-1])
                if dist <= eps:
                    dp[i][j] = dp[i-1][j-1] + 1
                    if return_alignment:
                        backtrack[i][j] = 0
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                    if return_alignment:
                        backtrack[i][j] = 1 if dp[i-1][j] >= dp[i][j-1] else 2

        # Compute similarity
        similarity = dp[m][n] / min(m, n)

        # Backtrack for alignment
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
    
    def compute_lcss_alignment(self, track1: List[GeoPoint], track2: List[GeoPoint]) -> List[Tuple[int, int]]:
        """Compute LCSS alignment between two tracks"""
        _, alignment = self._compute_lcss_core(
            track1, track2,
            eps=self.config.eps,
            window=self.config.lcss_window_size,
            return_alignment=True
        )
        return alignment or []
            
    def calculate_route_similarity(self, route1: Route, route2: Route) -> float:
        """Calculate similarity based on LCSS with auto-adjusted distance"""
        if len(route1.points) < 5 or len(route2.points) < 5:
            print_step("SIMILARITY", f"Skipping calc: routes too short ({len(route1.points)}, {len(route2.points)} pts)", "INFO")
            return 0.0
        
        # Auto-adjust eps based on route characteristics
        self.config.eps = self.calculate_eps_for_lcss(route1.points, route2.points)
        #print_step("SIMILARITY", f"Auto-tuned eps: {self.config.eps:.1f}m for routes '{route1.name}' & '{route2.name}'", "INFO")

        # Compute LCSS similarity
        similarity, _ = self._compute_lcss_core(
            route1.points, route2.points, 
            self.config.eps, self.config.lcss_window_size, 
            return_alignment=False
        )

        print_step("SIMILARITY", f"Similarity between '{route1.name}' and '{route2.name}': {similarity:.3f}", "INFO")
        return similarity

    def calculate_eps_for_lcss(
        self,
        track1: List[GeoPoint],
        track2: List[GeoPoint],
        multiplier: float = 1.8
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
    """Placeholder for hierarchical route merging - to be implemented"""
    
    def __init__(self, similarity_analyzer: RouteSimilarityAnalyzer):
        self.analyzer = similarity_analyzer
        print("Route merger initialized - advanced merging functionality pending implementation")

    def build_route_graph(self, routes: List[Route]) -> None:
        """Identify similar routes and prepare for merging"""
        if len(routes) < 2:
            print_step("RouteMerging", "Not enough routes to merge (need at least 2)")
            return
        
        print_step("RouteMerging", f"Analyzing {len(routes)} routes for similarity...")
        
        # Find route pairs that are similar enough to merge
        similar_pairs = []
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                similarity = self.analyzer.calculate_route_similarity(routes[i], routes[j])
                if similarity >= self.analyzer.config.similarity_threshold:
                    similar_pairs.append((routes[i], routes[j], similarity))
        
        # Group routes into merge clusters
        self.merge_clusters = self._group_routes_into_clusters(routes, similar_pairs)
        print_step("RouteMerging", f"Created {len(self.merge_clusters)} merge clusters")
                
    def _group_routes_into_clusters(self, routes: List[Route], similar_pairs: List) -> List[List[Route]]:
        """Group routes using connected components (transitive similarity)."""
        from collections import defaultdict, deque

        # Build adjacency graph using Route objects (now hashable!)
        graph = defaultdict(set)
        for route1, route2, _ in similar_pairs:
            graph[route1].add(route2)
            graph[route2].add(route1)

        visited = set()
        clusters = []

        for route in routes:
            if route in visited:
                continue
            if route not in graph:  # standalone route
                clusters.append([route])
                visited.add(route)
                print_step("RouteMerging", f"Route '{route.name}' will remain standalone")
                continue

            # BFS to find all connected routes (transitive closure)
            component = []
            queue = deque([route])
            while queue:
                r = queue.popleft()
                if r in visited:
                    continue
                visited.add(r)
                component.append(r)
                queue.extend(graph[r] - visited)

            clusters.append(component)
            print_step("RouteMerging", f"Created merged cluster with {len(component)} routes: {[r.name for r in component]}")

        return clusters

    def merge_routes(self) -> List[EstablishedRoute]:
        """Merge similar routes into EstablishedRoutes"""
        if not hasattr(self, 'merge_clusters') or not self.merge_clusters:
            print_step("RouteMerging", "No routes to merge")
            return []
        
        established_routes = []
        
        for cluster in self.merge_clusters:
            # Merge multiple routes
            merged_segments, std_devs = self._merge_route_cluster(cluster)
            original_routes = {route.name: route for route in cluster}
            
            rname = f"Merged ({cluster[0].name} etc.)" if len(cluster) > 1 else cluster[0].name
            
            established_route = EstablishedRoute.from_segments_and_stats(
                name=rname,
                segments=merged_segments,
                original_routes=original_routes,
                std_devs=std_devs
            )
            established_routes.append(established_route)
            print_step("RouteMerging", 
                    f"Created merged route from {len(cluster)} routes: '{established_route.name}'")
        
        print_step("RouteMerging", f"Created {len(established_routes)} established routes total")
        return established_routes

    def _merge_route_cluster(self, routes: List[Route]) -> Tuple[List[RouteSegment], List[Dict[str, float]]]:
        """Returns (segments, per-point standard deviations)"""
        if len(routes) == 1:
            return self._create_single_route_segment(routes[0])
        
        reference = self._select_reference_route(routes)
        point_groups = self._build_point_groups(reference, routes)        
        
        # Single pass: compute merged route AND detect transitions
        merged_points, std_devs, transitions = self._merge_to_reference_with_transitions(reference, point_groups)
        
        # Store transitions for Phase 2 (graph construction)
        self._detected_transitions = transitions
        
        segment = RouteSegment(
            points=merged_points,
            source_route_names={r.name for r in routes},
            confidence=0.8
        )
        
        return [segment], std_devs

    def _create_single_route_segment(self, route: Route) -> Tuple[List[RouteSegment], List[Dict[str, float]]]:
        """Create segment for a single route with zero standard deviation."""
        segment = RouteSegment(
            points=list(route.points),
            source_route_names={route.name},
            confidence=1.0
        )
        std_devs = [{"lat_std": 0.0, "lon_std": 0.0, "ele_std": 0.0} for _ in route.points]
        return [segment], std_devs

    def _select_reference_route(self, routes: List[Route]) -> Route:
        """Select the longest route as reference backbone."""
        return max(routes, key=lambda r: r.total_distance or 0)

    def _build_point_groups(self, reference: Route, routes: List[Route]) -> Dict[int, List[GeoPoint]]:
        """Build groups of aligned points from all routes to the reference."""
        point_groups = defaultdict(list)
        
        for other in routes:
            if other is reference:
                continue
            alignment = self.analyzer.compute_lcss_alignment(reference.points, other.points)
            for i_ref, i_other in alignment:
                point_groups[i_ref].append(other.points[i_other])
        
        return point_groups

    def _merge_to_reference_with_transitions(self, reference: Route, point_groups: Dict[int, List[GeoPoint]]) -> Tuple[
        List[GeoPoint], 
        List[Dict[str, float]], 
        List[RouteTransition]
    ]:
        """Single pass that computes merged route AND detects transitions."""
        merged_points = []
        std_devs = []
        transitions: List[RouteTransition] = []
        current_transition: Optional[RouteTransition] = None
        
        min_transition_length = 3
        confidence_level = self.analyzer.config.outlier_confidence_level
        
        for i_ref, ref_point in enumerate(reference.points):
            matches = point_groups.get(i_ref, [])
            
            if matches:
                total_points = [ref_point] + matches
                merged_point, std_dev, residuals = self._compute_median_consensus_with_residuals(total_points)
                outlier_count, consensus_count, max_residual_ratio = self._analyze_outliers(
                    residuals, std_dev, len(total_points), confidence_level
                )
                
                has_outliers = outlier_count > 0 and outlier_count < len(total_points)
                
                if has_outliers:
                    if current_transition is None:
                        current_transition = RouteTransition(
                            start_idx=i_ref,
                            start_point=ref_point,
                            end_idx=i_ref,
                            end_point=ref_point,
                            outlier_count=outlier_count,
                            consensus_count=consensus_count,
                            max_residual_ratio=max_residual_ratio
                        )
                    else:
                        current_transition.end_idx = i_ref
                        current_transition.end_point = ref_point
                        current_transition.max_residual_ratio = max(
                            current_transition.max_residual_ratio, 
                            max_residual_ratio
                        )
                else:
                    if current_transition:
                        transition_length = current_transition.end_idx - current_transition.start_idx + 1
                        if transition_length >= min_transition_length:
                            transitions.append(current_transition)
                        current_transition = None
                
                merged_points.append(merged_point)
                std_devs.append(std_dev)
            else:
                if current_transition:
                    transition_length = current_transition.end_idx - current_transition.start_idx + 1
                    if transition_length >= min_transition_length:
                        transitions.append(current_transition)
                    current_transition = None
                
                merged_points.append(ref_point)
                std_devs.append({"lat_std": 0.0, "lon_std": 0.0, "ele_std": 0.0})
        
        if current_transition:
            transition_length = current_transition.end_idx - current_transition.start_idx + 1
            if transition_length >= min_transition_length:
                transitions.append(current_transition)
        
        return merged_points, std_devs, transitions

    def _compute_median_consensus_with_residuals(self, points: List[GeoPoint]) -> Tuple[GeoPoint, Dict[str, float], List[Tuple[float, float]]]:
        """Compute median point, standard deviations, and residuals using standard library."""
        lats = [p.lat for p in points]
        lons = [p.lon for p in points]
        eles = [p.elevation for p in points]
        
        # Use statistics.median for robustness
        median_lat = median(lats)
        median_lon = median(lons)
        median_ele = median(eles)
        
        lat_std = stdev(lats) if len(lats) > 1 else 0.0
        lon_std = stdev(lons) if len(lons) > 1 else 0.0
        ele_std = stdev(eles) if len(eles) > 1 else 0.0
        
        # Calculate residuals for statistical testing
        lat_residuals = [abs(p.lat - median_lat) for p in points]
        lon_residuals = [abs(p.lon - median_lon) for p in points]
        residuals = list(zip(lat_residuals, lon_residuals))
        
        merged_point = GeoPoint(median_lat, median_lon, median_ele)
        std_dev = {"lat_std": lat_std, "lon_std": lon_std, "ele_std": ele_std}
        
        return merged_point, std_dev, residuals
    
    def _analyze_outliers(self, residuals: List[Tuple[float, float]], std_dev: Dict[str, float], n_points: int, confidence_level: float = 0.95) -> Tuple[int, int, float]:
        """
        Analyze residuals using Student's t-distribution.
        Returns (outlier_count, consensus_count, max_residual_ratio)
        """
        if n_points < 3:
            return 0, n_points, 0.0
        
        df = n_points - 1
        alpha = 1.0 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        max_lat_residual_allowed = t_critical * std_dev['lat_std']
        max_lon_residual_allowed = t_critical * std_dev['lon_std']
        
        outlier_count = 0
        max_residual_ratio = 0.0
        
        for lat_res, lon_res in residuals:
            if std_dev['lat_std'] > 0:
                lat_ratio = lat_res / max_lat_residual_allowed
            else:
                lat_ratio = 0.0 if lat_res == 0 else float('inf')
                
            if std_dev['lon_std'] > 0:
                lon_ratio = lon_res / max_lon_residual_allowed
            else:
                lon_ratio = 0.0 if lon_res == 0 else float('inf')
                
            max_ratio = max(lat_ratio, lon_ratio)
            
            if max_ratio > 1.0:
                outlier_count += 1
            
            max_residual_ratio = max(max_residual_ratio, max_ratio)
        
        consensus_count = n_points - outlier_count
        return outlier_count, consensus_count, max_residual_ratio        

# Utility function for easy integration
def create_route_merger(config: Optional[RouteSimilarityConfig] = None) -> HierarchicalRouteMerger:
    """Convenience function to create a configured route merger"""
    analyzer = RouteSimilarityAnalyzer(config)
    return HierarchicalRouteMerger(analyzer)