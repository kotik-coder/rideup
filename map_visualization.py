# map_visualization.py
import plotly.graph_objects as go
from typing import List, Optional, Dict
from route import GeoPoint
from route_processor import ProcessedRoute
from checkpoints import Checkpoint
from spot import Spot

def calculate_viewport(lats: List[float], lons: List[float], 
                     base_padding_ratio: float = 0.1,
                     target_aspect_ratio: float = 1.6) -> Dict[str, float]:
    """Calculate viewport bounds for map display."""
    if not lats or not lons:
        return {}

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # Calculate extents with fallbacks
    lat_extent = max_lat - min_lat or 0.001
    lon_extent = max_lon - min_lon or 0.001

    # Apply base padding
    lat_min = min_lat - lat_extent * base_padding_ratio
    lat_max = max_lat + lat_extent * base_padding_ratio
    lon_min = min_lon - lon_extent * base_padding_ratio
    lon_max = max_lon + lon_extent * base_padding_ratio

    # Adjust for aspect ratio
    current_ratio = (lon_max - lon_min) / (lat_max - lat_min)
    if current_ratio < target_aspect_ratio:
        required_lon_extent = (lat_max - lat_min) * target_aspect_ratio
        expansion = (required_lon_extent - (lon_max - lon_min)) / 2
        lon_min -= expansion
        lon_max += expansion
    elif current_ratio > target_aspect_ratio:
        required_lat_extent = (lon_max - lon_min) / target_aspect_ratio
        expansion = (required_lat_extent - (lat_max - lat_min)) / 2
        lat_min -= expansion
        lat_max += expansion

    return {'west': lon_min, 'east': lon_max, 'south': lat_min, 'north': lat_max}

def add_route_to_figure(fig: go.Figure, 
                       processed_route: ProcessedRoute,
                       selected_checkpoint_index: Optional[int] = None) -> None:
    """Add route visualization to map figure."""
    # Add route line
    fig.add_trace(go.Scattermap(
        mode="lines",
        lon=[p.lon for p in processed_route.smooth_points],
        lat=[p.lat for p in processed_route.smooth_points],
        line=dict(width=4, color="blue"),
        name=processed_route.route.name,
        showlegend=True
    ))

    # Add checkpoints
    checkpoint_colors = [
        'gold' if i == selected_checkpoint_index else 'green'
        for i in range(len(processed_route.checkpoints))
    ]
    fig.add_trace(go.Scattermap(
        mode="markers",
        lon=[c.lon for c in processed_route.checkpoints],
        lat=[c.lat for c in processed_route.checkpoints],
        marker=dict(size=12, color=checkpoint_colors),
        name="Checkpoints",
        showlegend=True
    ))

def create_base_map(spot: Spot) -> go.Figure:
    """Create initial map figure centered on spot and fitting its bounds."""
    fig = go.Figure()
    
    # Use spot.bounds directly for fitbounds within the mapbox dictionary
    min_lon, min_lat, max_lon, max_lat = spot.bounds

    print(spot.bounds)
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            # Set bounds directly here for fitbounds behavior
            bounds={
                'west': min_lon, 
                'east': max_lon, 
                'south': min_lat, 
                'north': max_lat
            }
        ),
        margin={"r":0,"t":0,"l":0,"b":0}
    )
        
    return fig

def show_available_routes(fig: go.Figure, spot: Spot):
    routes = spot.routes
    if routes:
        for r in routes:
            add_route_to_figure(fig, r)