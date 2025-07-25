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
        name = f"Route - {route.name}",
        customdata=[route_index] * len(points),  # Add route index to each point
        hoverinfo="name",  # Only show text in hover
        showlegend=True,
    ))
    
def hide_base_route(fig: go.Figure, 
                    spot : Spot,
                    route: Route):        
        
    for trace in fig.data:
        if trace.name == route.name:
            trace.visible = False
        elif trace.name != spot.name:            
            if trace.name and trace.name.startswith("Route -"):
                trace.line.color = "grey"            
                trace.line.width = 5
                trace.hoverinfo  = "name"
                trace.showlegend = True
            else:
                trace.visible = False
                trace.showlegend = False
                trace.hoverinfo = 'none'
                
def add_full_route_to_figure(fig: go.Figure, r: ProcessedRoute, route_profiles: dict, route : Route):
    """
    Adds a performant, multi-layered route visualization to a Plotly figure.
    - Groups segments by type to reduce the number of traces and improve performance.
    - Uses a shadow/base/highlight design for high contrast against map tiles.
    """
    # Visual properties
    highlight_width = 10
    base_width = 8
    shadow_width = 14
    
    # Pre-calculate full route coordinates to avoid repetition
    full_lon = [p.lon for p in r.smooth_points]
    full_lat = [p.lat for p in r.smooth_points]

    # 1. Shadow effect for depth and contrast
    fig.add_trace(go.Scattermap(
        lon=full_lon,
        lat=full_lat,
        mode="lines",
        line=dict(width=shadow_width, color="rgba(0, 0, 0, 0.4)"),
        hoverinfo="none",
        showlegend=False,
        name=f"shadow_{route.name}",
    ))

    # 2. White base route to sit on top of the shadow
    fig.add_trace(go.Scattermap(
        lon=full_lon,
        lat=full_lat,
        mode="lines",
        line=dict(width=base_width, color="rgba(255, 255, 255, 0.9)"),
        hoverinfo="none",
        showlegend=False,
        name=f"processed_{route.name}",
    ))

    # 3. Highlighted segments, refactored for performance
    if 'segments' in route_profiles and route_profiles['segments']:
        color_map = {
            ElevationSegmentType.ASCENT: "rgba(230, 50, 50, 0.95)",
            ElevationSegmentType.DESCENT: "rgba(50, 100, 230, 0.95)",
            ElevationSegmentType.STEEP_ASCENT: "rgba(255, 0, 0, 1.0)",
            ElevationSegmentType.STEEP_DESCENT: "rgba(0, 0, 255, 1.0)",
            ElevationSegmentType.ROLLER: "rgba(255, 165, 0, 0.95)",
            ElevationSegmentType.SWITCHBACK: "rgba(255, 215, 0, 1.0)"
        }
        
        # Group all points and hover data by their segment type
        traces_data = {}
        for segment in route_profiles['segments']:
            # Access segment_type, start_index, end_index, length, and gradients as dictionary keys
            # The seg_type from the dictionary will be the string name of the enum, e.g., "ASCENT"
            seg_type_str = segment['segment_type'] 
            # Convert string back to ElevationSegmentType enum member for the color_map lookup
            seg_type = ElevationSegmentType[seg_type_str] 
            
            start_index = segment['start_index']
            end_index = segment['end_index']
            segment_length = segment['length']

            if seg_type not in color_map or segment_length < 50:
                continue

            if seg_type not in traces_data:
                traces_data[seg_type] = {"lon": [], "lat": [], "hovertext": []}
            
            segment_points = r.smooth_points[start_index:end_index + 1] 
            hover_info = (
                f"<b>{seg_type.name.replace('_', ' ').title()}</b><br>"
                f"Length: {segment_length:.0f} m<br>"
                f"Avg Grade: {segment['avg_gradient'] * 100:.1f} % <br>"
                f"Max Grade: {segment['max_gradient'] * 100:.1f} %"
            )
            
            for p in segment_points:
                traces_data[seg_type]['lon'].append(p.lon)
                traces_data[seg_type]['lat'].append(p.lat)
                traces_data[seg_type]['hovertext'].append(hover_info)
            
            traces_data[seg_type]['lon'].append(None)
            traces_data[seg_type]['lat'].append(None)
            traces_data[seg_type]['hovertext'].append(None)

        for seg_type, data in traces_data.items():
            fig.add_trace(go.Scattermap(
                lon=data['lon'],
                lat=data['lat'],
                mode="lines",
                line=dict(width=highlight_width, color=color_map[seg_type]),
                hovertext=data['hovertext'],
                hoverinfo="text",
                name=seg_type.name.replace('_', ' ').title(),
                legendgroup="segments",
            ))

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
                color='gold',
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
                                  profiles : dict) -> go.Figure:
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
        
    add_full_route_to_figure(fig, processed_route, profiles, route)
    
    print_step("Map Drawing", f"Карта обновлена для выбранного маршрута: {route.name}")
    return fig