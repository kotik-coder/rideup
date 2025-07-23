# map_visualization.py
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple

from src.ui.map_helpers import bounds_to_zoom, print_step
from src.routes.route import Route
from src.routes.route_processor import ProcessedRoute
from src.routes.spot import Spot
from src.routes.statistics_collector import ElevationSegmentType, StaticProfilePoint

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
    
def add_full_route_to_figure(fig: go.Figure, r: ProcessedRoute, route_profiles: dict):
    """Adds route visualization that contrasts with map features"""
    # Visual properties designed to contrast with map colors
    base_width = 8
    highlight_width = 10
    shadow_width = 14
    shadow_color = "rgba(0, 0, 0, 0.4)"  # Darker shadow for better contrast
    base_color = "rgba(255, 255, 255, 0.9)"  # Bright white base with high opacity
    
    # Color mapping using colors that contrast with typical map features
    color_map = {
        ElevationSegmentType.ASCENT: "rgba(230, 50, 50, 0.95)",       # Vivid red
        ElevationSegmentType.DESCENT: "rgba(50, 100, 230, 0.95)",     # Deep blue
        ElevationSegmentType.STEEP_ASCENT: "rgba(255, 0, 0, 1.0)",    # Pure red
        ElevationSegmentType.STEEP_DESCENT: "rgba(0, 0, 255, 1.0)",   # Pure blue
        ElevationSegmentType.ROLLER: "rgba(255, 165, 0, 0.95)",       # Orange
        ElevationSegmentType.SWITCHBACK: "rgba(255, 215, 0, 1.0)"     # Bright gold
    }

    # 1. Shadow effect - now more pronounced
    fig.add_trace(go.Scattermap(
        mode="lines",
        lon=[p.lon for p in r.smooth_points],
        lat=[p.lat for p in r.smooth_points],
        line=dict(width=shadow_width, color=shadow_color),
        hoverinfo="none",
        showlegend=False
    ))

    # 2. Base route - now white with dark outline for maximum contrast
    fig.add_trace(go.Scattermap(
        mode="lines",
        lon=[p.lon for p in r.smooth_points],
        lat=[p.lat for p in r.smooth_points],
        line=dict(
            width=base_width, 
            color=base_color,
        ),
        hoverinfo="none",
        showlegend=False
    ))

    # 3. Highlighted segments with strong contrasting colors
    if 'segments' in route_profiles and route_profiles['segments']:
        for segment in route_profiles['segments']:
            if segment.segment_type not in color_map or segment.length < 50:
                continue
                
            segment_points = r.smooth_points[segment.start_index:segment.end_index+1]
            
            fig.add_trace(go.Scattermap(
                mode="lines",
                lon=[p.lon for p in segment_points],
                lat=[p.lat for p in segment_points],
                line=dict(
                    width=highlight_width,
                    color=color_map[segment.segment_type],
                ),
                hovertext=(
                    f"{segment.segment_type.name}\n"
                    f"Length: {segment.length:.0f}m\n"
                    f"Avg Gradient: {segment.avg_gradient*100:.1f}%\n"
                    f"Max Gradient: {segment.max_gradient*100:.1f}%"
                ),
                hoverinfo="text",
                showlegend=False
            ))

    # 4. Checkpoints (topmost layer)
    add_checkpoints(fig, r)

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
            color='black',
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
                                  map_dims : Dict[str,int],
                                  profiles : List[StaticProfilePoint]) -> go.Figure:
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
        
    add_full_route_to_figure(fig, processed_route, profiles)
    
    print_step("Map Drawing", f"Карта обновлена для выбранного маршрута: {route.name}")
    return fig