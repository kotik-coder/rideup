# map_visualization.py
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple

from src.ui.map_helpers import bounds_to_zoom, print_step
from src.routes.route import Route
from src.routes.route_processor import ProcessedRoute
from src.routes.spot import Spot

intermediate_points_label = "Intermediate"
checkpoints_label = "Checkpoint"

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
    
def hide_base_route(fig: go.Figure, 
                    spot : Spot,
                    route: Route):        
        
    for trace in fig.data:
        if trace.name == route.name and trace.mode == "lines":
            trace.line.color = "rgba(150, 150, 150, 0.3)"
            trace.hoverinfo = "none"
            trace.showlegend = False
        elif trace.name != spot.name:
            trace.line.color = "grey"
            trace.hoverinfo  = "name", 
            trace.showlegend = True                    
    
def add_full_route_to_figure(fig: go.Figure,
                       r : ProcessedRoute):

    # Trace for all smooth points as non-selectable markers (intermediate points)
    non_checkpoint_lons = []
    non_checkpoint_lats = []
    non_checkpoint_elevations = []
    checkpoint_indices = {cp.route_point_index for cp in r.checkpoints}

    for i, p in enumerate(r.smooth_points):
        if i not in checkpoint_indices:
            non_checkpoint_lons.append(p.lon)
            non_checkpoint_lats.append(p.lat)
            non_checkpoint_elevations.append(p.elevation)

    if non_checkpoint_lons:
        fig.add_trace(go.Scattermap(
            mode="markers",
            lon=non_checkpoint_lons,
            lat=non_checkpoint_lats,
            marker=dict(
                size=10,
                color=non_checkpoint_elevations,
                colorscale='hot',
                opacity=1.0,  # Force full opacity
                colorbar=dict(
                    title='Высота (м)',
                    x=0.01,  # Move more to the left
                    y=0.85,   # Position above the graphs
                    xanchor='left',
                    yanchor='top',
                    len=0.3,  # Make it shorter
                    thickness=15,  # Make it narrower
                    title_font=dict(
                        size=12,
                        color='black'
                    ),
                    tickfont=dict(
                        size=10,
                        color='black'
                    ),
                    bgcolor='rgba(255,255,255,0.5)',  # Semi-transparent white background
                    outlinecolor='black',
                    outlinewidth=1
                ),
            ),
            hoverinfo="none",
            showlegend=False,
            name=intermediate_points_label,
            selected=dict(marker=dict(opacity=1.0)),  # Disable selection effects
            unselected=dict(marker=dict(opacity=1.0))
        ))

def add_checkpoints(fig: go.Figure, r: ProcessedRoute):
    checkpoints = r.checkpoints
    
    # Ensure checkpoints are sorted by their point_index
    checkpoints = sorted(checkpoints, key=lambda cp: cp.checkpoint_index)
    
    checkpoint_lons = [cp.point.lon for cp in checkpoints]
    checkpoint_lats = [cp.point.lat for cp in checkpoints]    
    checkpoint_customdata = [
        [idx, 
         cp.name, 
         cp.description, 
         cp.point.elevation, 
         cp.distance_from_origin]
        for idx, cp in enumerate(checkpoints)  # Use enumerate to ensure sequential indexing
    ]

    fig.add_trace(go.Scattermap(
        mode="markers",
        lon=checkpoint_lons,
        lat=checkpoint_lats,
        marker=dict(
            size=12,
            color='green',
            opacity=0.5,
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
        selected=dict(
            marker=dict(
                size=15,
                color='darkgreen',
                opacity=1.0
            )
        ),
        unselected=dict(
            marker=dict(opacity=0.5)
        ),
        showlegend=True,
        name=checkpoints_label
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
            bounds=square,
            center=dict(lat=center_lat, lon=center_lon),
        ),        
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=False,
        clickmode='event+select'
    )
    
def zoom_on_feature(fig : go.Figure, bounds : List[float], width : int , height : int):    
    
    if len(bounds) != 4:
        print_step("Map", "Invalid bounds when centering!")
    
    lon_min, lat_min, lon_max, lat_max = bounds
    
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    
    bounds_dict = {"max_lon" : lon_max,
                   "min_lon" : lon_min,
                   "max_lat" : lat_max,
                   "min_lat" : lat_min}
    
    calculated_zoom = bounds_to_zoom(bounds_dict, width, height, padding = 0.05)
    
    fig.update_layout(
        map_style="open-street-map",
        map=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=calculated_zoom,),
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=False,
        clickmode='event+select'
    )

def update_map_for_selected_route(current_map_figure: dict, 
                                  spot: Spot,  
                                  route: Route,                                 
                                  processed_route: ProcessedRoute,
                                  map_dims : Dict[str,int]) -> go.Figure:
    """
    Clears existing route traces and plots the fully processed route.
    Returns the updated figure.
    """
    fig = go.Figure(current_map_figure)
    hide_base_route(fig, spot, route)

    traces_to_keep = []
    to_delete = [intermediate_points_label, checkpoints_label]
    for trace in fig.data:
        if isinstance(trace, go.Scattermap):
            if trace.name not in to_delete:
                traces_to_keep.append(trace)
    
    fig.data = traces_to_keep    
    
    # Use the dimensions to zoom correctly instead of just centering.
    if map_dims:
        height = map_dims['height']
        if height < 1:
            height = int( map_dims['width'] * 9/16 )
        zoom_on_feature(fig, processed_route.bounds, map_dims['width'], height)
    else:
        # Fallback if dimensions aren't ready yet
        center_on_feature(fig, processed_route.bounds)
        
    add_full_route_to_figure(fig, processed_route)
    
    print_step("Map Drawing", f"Карта обновлена для выбранного маршрута: {route.name}")
    return fig