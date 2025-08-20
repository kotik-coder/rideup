# map_visualization.py
import plotly.graph_objects as go
from typing import Dict, List

from src.routes.profile_analyzer import Profile
from src.ui.map_helpers import bounds_to_zoom, print_step
from src.routes.route import Route
from src.routes.route_processor import ProcessedRoute
from src.routes.spot import Spot
from src.ui.trail_style import COLOR_MAP, WIDTH_CONFIG

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
                

def add_full_route_to_figure(fig: go.Figure, r: ProcessedRoute, profile : Profile, route: Route):
    """
    Adds a performant, multi-layered route visualization to a Plotly figure.
    - Uses Scattermap instead of Scattermapbox
    - Maintains feature highlighting within Scattermap's capabilities
    - Groups segments by type for performance
    """
    # Use configurations from trail_style.py
    color_map = COLOR_MAP
    width_config = WIDTH_CONFIG

    # Pre-calculate full route coordinates
    full_lon = [p.lon for p in r.smooth_points]
    full_lat = [p.lat for p in r.smooth_points]

    # 1. Shadow effect
    fig.add_trace(go.Scattermap(
        lon=full_lon,
        lat=full_lat,
        mode="lines",
        line=dict(width=width_config['shadow'], color="rgba(0, 0, 0, 0.4)"),
        hoverinfo="none",
        showlegend=False,
        name=f"shadow_{route.name}",
    ))

    # 2. Base route
    fig.add_trace(go.Scattermap(
        lon=full_lon,
        lat=full_lat,
        mode="lines",
        line=dict(width=width_config['base'], color="rgba(255, 255, 255, 0.9)"),
        hoverinfo="none",
        showlegend=False,
        name=f"base_{route.name}",
    ))

    # 3. Highlighted segments
    segment_profile = profile.segments
    traces_data = {}
    
    for segment in segment_profile:
        if segment.feature:        
            ftype = segment.feature.feature_type
            if ftype in color_map:
                seg_type = ftype
        elif segment.gradient_type in color_map:
            seg_type = segment.gradient_type
        else:
            continue
        
        start_idx = segment.start_index
        end_idx = segment.end_index
        
        if seg_type not in traces_data:
            traces_data[seg_type] = {
                'lon': [],
                'lat': [],
                'hovertext': [],
                'count': 0
            }
        
        segment_points = r.smooth_points[start_idx:end_idx + 1]
        
        ttl = seg_type.title()
        
        # Build comprehensive hover info
        hover_info = (
            f"<b>{ttl}</b><br>"
            f"Length: {segment.length(profile.points):.0f}m<br>"
            f"Grade: {segment.grade(profile.points)*100:.1f}%<br>"
            f"Max gradient: {segment.max_gradient(profile.points)*100:.1f}%"
        )
        
        # Add short features if present
        if segment.short_features:
            hover_info += "<br><br><b>Short Features:</b>"
            for feature in segment.short_features:
                ftype = feature.feature_type
                print(ftype.title())
                hover_info += (
                    f"<br>• {ftype.title()}: "
                    f"{feature.len:.1f}m, {feature.grade*100:.1f}%"
                )
        
        # Add segment points
        traces_data[seg_type]['lon'].extend(p.lon for p in segment_points)
        traces_data[seg_type]['lat'].extend(p.lat for p in segment_points)
        traces_data[seg_type]['hovertext'].extend([hover_info] * len(segment_points))
        traces_data[seg_type]['count'] += 1
        
        # Add break between segments
        traces_data[seg_type]['lon'].append(None)
        traces_data[seg_type]['lat'].append(None)
        traces_data[seg_type]['hovertext'].append(None)

    # Add traces with width variations
    for seg_type, data in traces_data.items():
        if data['count'] == 0:
            continue
            
        fig.add_trace(go.Scattermap(
            lon=data['lon'],
            lat=data['lat'],
            mode="lines",
            line=dict(
                width=width_config['highlight'].get(seg_type.name, width_config['highlight']['default']),
                color=color_map[seg_type]
            ),
            hovertext=data['hovertext'],
            hoverinfo="text",
            name=seg_type.name.replace('_', ' ').title(),
            legendgroup="segments",
            showlegend=True
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
                                  profile : Profile) -> go.Figure:
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
        
    add_full_route_to_figure(fig, processed_route, profile, route)
    
    print_step("Map Drawing", f"Карта обновлена для выбранного маршрута: {route.name}")
    return fig