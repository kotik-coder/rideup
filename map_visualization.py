# map_visualization.py
import osmnx as ox
import plotly.graph_objects as go
from typing import List, Optional, Dict, Tuple

from shapely import Polygon
from map_helpers import print_step
from route import GeoPoint, Route
from route_processor import ProcessedRoute
from checkpoints import Checkpoint
from spot import Spot


def plot_available_routes(fig: go.Figure, spot: Spot):
    routes = spot.routes
    if routes:
        for r in routes:
            add_base_route_to_figure(fig, r)

def add_base_route_to_figure(fig: go.Figure, 
                       route):
    """Add route visualization to map figure."""
    
    if type(route) == ProcessedRoute:
        points = route.smooth_points
        name = route.route.name
    else:
        points = route.points
        name = route.name
    
    # Add route line
    fig.add_trace(go.Scattermap(
        mode="lines",
        lon=[p.lon for p in points],
        lat=[p.lat for p in points],
        line=dict(width=3, color="grey"),
        name=name,
        hoverinfo="name",  # Only show text in hover
        showlegend=True
    ))
    
def add_full_route_to_figure(fig: go.Figure, 
                       r : ProcessedRoute,
                       selected_checkpoint_index: Optional[int] = None): 
    
    add_base_route_to_figure(fig, r)

    # Add checkpoints
    checkpoint_colors = [
        'gold' if i == selected_checkpoint_index else 'green'
        for i in range(len(r.checkpoints))
    ]
    fig.add_trace(go.Scattermap(
        mode="markers",
        lon=[c.lon for c in r.checkpoints],
        lat=[c.lat for c in r.checkpoints],
        marker=dict(size=12, color=checkpoint_colors),
        hoverinfo="name",  # Only show text in hover
        name="Checkpoints",
        showlegend=True
    ))
    
def _add_spot_boundary(fig, spot : Spot):
    """Adds the forest boundary and name annotation to the Plotly figure."""
    spot_lons, spot_lats = zip(*spot.geometry)
    fig.add_trace(go.Scattermap(
        lat=spot_lats,
        lon=spot_lons,
        mode="lines",
        fill="toself",
        fillcolor="rgba(0, 255, 0, 0.2)",
        line=dict(color="green", width=2),
        hoverinfo="name",  # Only show text in hover
        name=spot.name
    ))

    print_step("Map Drawing", "Добавлены границы и название леса.")
    
def create_base_map(spot : Spot):
    """Creates the base Plotly map figure with centering and style."""
    lon_min, lat_min, lon_max, lat_max = spot.bounds

    # Calculate center, though with 'bounds' Plotly will adjust
    # to center the given bounds, so this becomes less critical for centering.
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    
    max_dim = max(lat_max - lat_min, lon_max - lon_min) * 1.5
    square=dict(
                west=center_lon  - max_dim,
                east=center_lon  + max_dim,
                south=center_lat - max_dim,
                north=center_lat + max_dim
            )
    
    fig = go.Figure(go.Scattermap())
    fig.update_layout(
        map_style="open-street-map",
        map=dict(
            bounds=square
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=False,
        clickmode='event+select'
    )          

    # It's important that this function adds traces to the figure, as
    # 'fitbounds' or the 'bounds' property in 'layout.map' relies on the
    # presence of data to calculate the extent.
    _add_spot_boundary(fig, spot)
    plot_available_routes(fig, spot)

    return fig