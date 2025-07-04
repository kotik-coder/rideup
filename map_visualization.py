# map_visualization.py
import plotly.graph_objects as go
import numpy as np
from typing import List, Optional
from map_helpers import print_step
from route import GeoPoint
from route_processor import ProcessedRoute
from checkpoints import Checkpoint
from spot import Spot

def calculate_zoom(lats: List[float], lons: List[float]) -> int:
    """Calculate zoom level based on geographic coverage."""
    if not lats or not lons:
        print_step("Zoom", "Расчет зума: Нет координат. Возвращаю дефолтный зум.")
        return 12

    lat_span = max(lats) - min(lats)
    lon_span = max(lons) - min(lons)

    if lat_span == 0 and lon_span == 0:
        print_step("Zoom", "Расчет зума: Нулевой охват. Возвращаю высокий зум.")
        return 15

    zoom_lat = 9.5 - np.log2(lat_span + 1e-6)
    zoom_lon = 9.5 - np.log2(lon_span + 1e-6)

    final_zoom = min(zoom_lat, zoom_lon, 18)
    print_step("Zoom", f"Рассчитан зум: {final_zoom:.2f}")
    return final_zoom

def add_spot_boundary_to_figure(fig: go.Figure, spot: Spot) -> None:
    """Add spot boundary and name annotation to the Plotly figure."""
    min_lon, min_lat, max_lon, max_lat = spot.bounds

    # Add rectangle for spot boundary
    fig.add_trace(go.Scattermap(
        mode="lines",
        lon=[min_lon, max_lon, max_lon, min_lon, min_lon],
        lat=[min_lat, min_lat, max_lat, max_lat, min_lat],
        marker={'size': 10, 'color': "red"},
        name="Spot Boundary",
        hoverinfo='text',
        hovertext=f"Граница спота: {spot.name}",
        showlegend=False
    ))

    # Add annotation for spot name
    fig.add_annotation(
        x=(min_lon + max_lon) / 2,
        y=max_lat,
        text=f"<b>{spot.name}</b>",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        yref="paper",
        yanchor="bottom"
    )

def add_route_to_figure(
    fig: go.Figure,
    smooth_points: List[GeoPoint],
    checkpoints: List[Checkpoint],
    is_selected: bool,
    highlight_checkpoint: Optional[int] = None
) -> None:
    """
    Adds a route to the Plotly figure, with different styles for selected/unselected.
    
    Args:
        fig (go.Figure): The Plotly figure to add the route to.
        smooth_points (List[GeoPoint]): The smoothed geographical points of the route.
        checkpoints (List[Checkpoint]): The checkpoints along the route.
        is_selected (bool): True if this is the currently selected route, False otherwise.
        highlight_checkpoint (Optional[int]): The index of the checkpoint to highlight, if any.
    """
    if not smooth_points:
        return

    route_lats = [p.lat for p in smooth_points]
    route_lons = [p.lon for p in smooth_points]
    route_elevations = [p.elevation for p in smooth_points]

    if is_selected:
        # Add the main route trace (selected style)
        fig.add_trace(go.Scattermap(
            mode="lines",
            lon=route_lons,
            lat=route_lats,
            marker={'size': 1},
            line=dict(width=5, color='blue'),
            hoverinfo='text',
            hovertext=[f"Lat: {p.lat:.4f}<br>Lon: {p.lon:.4f}<br>Elevation: {p.elevation:.1f}m" for p in smooth_points],
            showlegend=False,
            name="Selected Route"
        ))

        # Add checkpoints as markers on the selected route
        if checkpoints:
            cp_lats = [c.lat for c in checkpoints]
            cp_lons = [c.lon for c in checkpoints]
            cp_names = [c.name for c in checkpoints]
            cp_distances = [c.distance_from_start for c in checkpoints]

            marker_colors = ['green'] * len(checkpoints)
            marker_sizes = [10] * len(checkpoints)
            marker_line_width = [1] * len(checkpoints)
            marker_line_color = ['black'] * len(checkpoints)

            if highlight_checkpoint is not None and 0 <= highlight_checkpoint < len(checkpoints):
                marker_colors[highlight_checkpoint] = 'gold'
                marker_sizes[highlight_checkpoint] = 12
                marker_line_width[highlight_checkpoint] = 2
                marker_line_color[highlight_checkpoint] = 'black'

            fig.add_trace(go.Scattermap(
                mode="markers",
                lon=cp_lons,
                lat=cp_lats,
                marker={
                    'size': marker_sizes,
                    'color': marker_colors,
                    'opacity': 1.0,
                    'allowoverlap': True,
                    'symbol': 'circle'
                },
                text=cp_names,
                hoverinfo='text',
                hovertext=[f"Чекпоинт: {name}<br>Дистанция: {dist:.1f}м" 
                        for name, dist in zip(cp_names, cp_distances)],
                customdata=list(range(len(checkpoints))),
                showlegend=False,
                name="Checkpoints"
            ))
    else:
        # Non-selected route gets simple gray line
        fig.add_trace(go.Scattermap(
            lat=route_lats,
            lon=route_lons,
            mode='lines',
            line=dict(width=3, color='rgba(100, 100, 100, 0.5)'),
            hoverinfo='text',
            hovertext=f"Маршрут: {smooth_points[0].lat:.4f}, {smooth_points[0].lon:.4f} (first point)",
            showlegend=False,
            name="Unselected Route"
        ))

def create_base_map(spot: Spot, routes: List[ProcessedRoute], selected_route_index: int = 0) -> go.Figure:
    """Creates a base map with spot boundaries and all routes."""
    fig = go.Figure()

    # Add spot boundaries
    add_spot_boundary_to_figure(fig, spot)

    # Add all routes (non-selected style)
    for i, route in enumerate(routes):
        add_route_to_figure(
            fig,
            route.smooth_points,
            route.checkpoints,
            is_selected=(i == selected_route_index)
        )

    # Set map layout
    all_lats = []
    all_lons = []
    for route in routes:
        all_lats.extend([p.lat for p in route.smooth_points])
        all_lons.extend([p.lon for p in route.smooth_points])

    if all_lats and all_lons:
        center_lat = np.mean(all_lats)
        center_lon = np.mean(all_lons)
        zoom = calculate_zoom(all_lats, all_lons)
    else:
        min_lon, min_lat, max_lon, max_lat = spot.bounds
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        zoom = calculate_zoom([min_lat, max_lat], [min_lon, max_lon])

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(
                lat=center_lat,
                lon=center_lon
            ),
            zoom=zoom
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        hovermode="closest"
    )

    return fig