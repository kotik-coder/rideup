# map_visualization.py
import numpy as np
import plotly.graph_objects as go
from typing import List, Optional, Tuple

from src.ui.map_helpers import print_step
from src.routes.route import Route
from src.routes.route_processor import ProcessedRoute
from src.routes.spot import Spot

intermediate_points_label = "Intermediate"
checkpoints_label = "Checkpoints"

def plot_available_routes(fig: go.Figure, spot: Spot):
    routes = spot.routes
    if routes:
        for i, r in enumerate(routes):
            add_base_route_to_figure(fig, r, route_index=i)  # Pass route index


def add_base_route_to_figure(fig: go.Figure, 
                           route: Route,
                           route_index: int):  # Add route_index parameter
    """Add route visualization to map figure."""
    
    points = route.points
    
    # Add route line with customdata containing route index
    fig.add_trace(go.Scattermap(
        mode="lines",
        lon=[p.lon for p in points],
        lat=[p.lat for p in points],
        line=dict(width=5, color="grey"),  # Make line thicker for easier clicking
        name = route.name,
        customdata=[route_index] * len(points),  # Add route index to each point
        hoverinfo="name",  # Only show text in hover
        showlegend=True
    ))
    
def add_full_route_to_figure(fig: go.Figure,
                       r : ProcessedRoute,
                       selected_checkpoint_index: Optional[int] = None):

    # Trace for all smooth points as non-selectable markers (intermediate points)
    non_checkpoint_lons = []
    non_checkpoint_lats = []
    non_checkpoint_elevations = []
    checkpoint_indices = {cp.point_index for cp in r.checkpoints}

    for i, p in enumerate(r.smooth_points):
        if i not in checkpoint_indices:
            non_checkpoint_lons.append(p.lon)
            non_checkpoint_lats.append(p.lat)
            non_checkpoint_elevations.append(p.elevation)

    if non_checkpoint_lons:
        fig.add_trace(go.Scattermap(
            mode="markers+lines",
            lon=non_checkpoint_lons,
            lat=non_checkpoint_lats,
            marker=dict(
                size=10,
                color=non_checkpoint_elevations,
                colorscale='hot',
                opacity=0.3,  # More translucent by default
                colorbar=dict(
                    title='Высота (м)',
                    x=0.05,
                    xanchor='left',
                    len=0.75,
                    thickness=20
                ),
            ),
            line=dict(width=3, color="grey"),
            hoverinfo="none",
            showlegend=False,
            name=intermediate_points_label,
        ))

    checkpoints = r.checkpoints
    
    # Prepare checkpoint data
    checkpoint_lons = [r.smooth_points[cp.point_index].lon for cp in checkpoints]
    checkpoint_lats = [r.smooth_points[cp.point_index].lat for cp in checkpoints]    
    checkpoint_customdata = [
        [i, cp.name, cp.description, cp.elevation, cp.distance_from_start]
        for i, cp in enumerate(checkpoints)
    ]

    # Create a list of marker sizes - larger for selected checkpoint
    marker_sizes = [15 if i == selected_checkpoint_index else 8 for i in range(len(checkpoints))]
    marker_opacities = [1.0 if i == selected_checkpoint_index else 0.7 for i in range(len(checkpoints))]

    fig.add_trace(go.Scattermap(
        mode="markers",
        lon=checkpoint_lons,
        lat=checkpoint_lats,
        marker=dict(
            size=marker_sizes,
            color='black',
            opacity=marker_opacities,
            symbol='circle',
        ),
        customdata=checkpoint_customdata,
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>" +
            "Широта: %{lat:.5f}<br>" +
            "Долгота: %{lon:.5f}<br>" +
            "Высота: %{customdata[3]:.1f} м<br>" +
            "Расстояние: %{customdata[4]:.1f} м<br>" +
            "%{customdata[2]}<extra></extra>"
        ),
        showlegend=True,
        name=checkpoints_label,
        selected=dict(
        marker=dict(
            size=12,
            color='white',            
            opacity=1.0,  # More translucent by default                
        ),
    ),
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
        hoverinfo="name",
        name=spot.name
    ))

    print_step("Map Drawing", "Добавлены границы и название леса.")
    
def create_base_map(spot : Spot):    
    fig = go.Figure(go.Scattermap())
    
    center_on_feature(fig, spot.bounds)

    _add_spot_boundary(fig, spot)
    plot_available_routes(fig, spot)

    # Configure hover behavior for the figure
    fig.update_layout(
        hovermode="closest",
        hoverdistance=10
    )

    return fig

def center_on_feature(fig : go.Figure, bounds : List[float]):    
    
    if len(bounds) != 4:
        print_step("Map", "Invalid bounds when centering!")
    
    lon_min, lat_min, lon_max, lat_max = bounds
    
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    
    max_dim = max(lat_max - lat_min, lon_max - lon_min) * 1.5
    square=dict(
                west=center_lon  - max_dim,
                east=center_lon  + max_dim,
                south=center_lat - max_dim,
                north=center_lat + max_dim
            )
    
    fig.update_layout(
        map_style="open-street-map",
        map=dict(
            bounds=square
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=False,
        clickmode='event+select'
    )

def update_map_for_selected_route(current_map_figure: dict, spot: Spot, processed_route: ProcessedRoute, selected_checkpoint_index: Optional[int] = None) -> go.Figure:
    """
    Clears existing route traces and plots the fully processed route.
    Returns the updated figure.
    """
    fig = go.Figure(current_map_figure)

    traces_to_keep = []
    to_delete = [intermediate_points_label, checkpoints_label]
    for trace in fig.data:
        if isinstance(trace, go.Scattermap):
            if trace.name not in to_delete and trace.name is not processed_route.route.name:
                traces_to_keep.append(trace)
    
    fig.data = traces_to_keep
    
    lons = [p.lon for p in processed_route.route.points]
    lats = [p.lat for p in processed_route.route.points]    
    
    center_on_feature(fig, [np.min(lons), np.min(lats), np.max(lons), np.max(lats)])
    add_full_route_to_figure(fig, processed_route, selected_checkpoint_index)
    
    print_step("Map Drawing", f"Карта обновлена для выбранного маршрута: {processed_route.route.name}")
    return fig