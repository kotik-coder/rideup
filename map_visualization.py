# map_visualization.py
import numpy as np
import plotly.graph_objects as go
from typing import List, Optional, Tuple

from map_helpers import print_step
from route import Route
from route_processor import ProcessedRoute
from spot import Spot


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
    name = route.name
    
    # Add route line with customdata containing route index
    fig.add_trace(go.Scattermap(
        mode="lines",
        lon=[p.lon for p in points],
        lat=[p.lat for p in points],
        line=dict(width=3, color="grey"),
        name=name,
        customdata=[route_index] * len(points),  # Add route index to each point
        hoverinfo="name",  # Only show text in hover
        showlegend=True
    ))
    
def add_full_route_to_figure(fig: go.Figure,
                       r : ProcessedRoute,
                       selected_checkpoint_index: Optional[int] = None):

    fig.add_trace(go.Scattermap(
        mode="markers+lines",  # Show both markers and connecting lines
        lon=[p.lon for p in r.smooth_points],
        lat=[p.lat for p in r.smooth_points],
        marker=dict(
            size=11,
            color=[p.elevation for p in r.smooth_points],
            colorscale='jet',
            showscale=True,
            colorbar=dict(
                title='Elevation (m)', # Clearer title
                x=0.05, # Position color bar on the left
                xanchor='left',
                len=0.75, # Adjust length
                thickness=20 # Adjust thickness
            ),
            opacity = 0.7
        ),
        line=dict(
            width=2,
            color="rgba(128,128,128,0.5)" # Subtle grey connecting lines (semi-transparent)
        ),
        hoverinfo="none", # No hover info for the route line itself
        showlegend=False
    ))

    # Add checkpoints as markers
    checkpoint_lons  = [cp.lon for cp in r.checkpoints]
    checkpoint_lats  = [cp.lat for cp in r.checkpoints]
    checkpoint_names = [cp.name for cp in r.checkpoints] # Labels are checkpoint names
    checkpoint_elevations = [cp.elevation for cp in r.checkpoints] # Labels are checkpoint names

    fig.add_trace(go.Scattermap(
        mode="markers+text",
        lon=checkpoint_lons,
        lat=checkpoint_lats,
        marker=dict(
            size=14,
            color="grey",
            symbol="circle"
        ),
        text=checkpoint_names,
        textposition="top right",
        textfont=dict(
            family="Arial",
            size=12,
            color="black"
        ),
        opacity=0.5,
        hovertext=[f"{name}<br>Высота: {elev:.1f}" for name, elev in zip(checkpoint_names, checkpoint_elevations)],
        hoverinfo="text",
        name="Контрольные точки", # Name for legend
        showlegend=True
    ))

    if selected_checkpoint_index is not None:
        selected_checkpoint = r.checkpoints[selected_checkpoint_index]
        fig.add_trace(go.Scattermap(
            mode="markers",
            lon=[selected_checkpoint.lon],
            lat=[selected_checkpoint.lat],
            marker=dict(
                size=14,
                color="#FFD700",  # Gold color for selected
                symbol="circle",
            ),
            opacity = 0.9,
            text=[selected_checkpoint.name],
            textposition="bottom center",
            textfont=dict(
                family="Arial",
                size=12,
                color="black",
                weight="bold"
            ),
            hoverinfo="text",
            hovertext=[f"Выбрана: {selected_checkpoint.name}<br>Высота: {selected_checkpoint.elevation}m"],
            name="Selected Checkpoint",
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
    fig = go.Figure(go.Scattermap())
    
    center_on_feature(fig, spot.bounds)

    # It's important that this function adds traces to the figure, as
    # 'fitbounds' or the 'bounds' property in 'layout.map' relies on the
    # presence of data to calculate the extent.
    _add_spot_boundary(fig, spot)
    plot_available_routes(fig, spot)

    return fig

def center_on_feature(fig : go.Figure, bounds : List[float]):    
    
    if len(bounds) != 4:
        print_step("Map", "Invalid bounds when centering!")
    
    lon_min, lat_min, lon_max, lat_max = bounds
    
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
    
    fig.update_layout(
        map_style="open-street-map",
        map=dict(
            bounds=square
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=False,
        clickmode='event+select'
    )

def update_map_for_selected_route(current_map_figure: dict, spot: Spot, processed_route: ProcessedRoute) -> go.Figure:
    """
    Clears existing route traces and plots the fully processed route.
    Returns the updated figure.
    """
    fig = go.Figure(current_map_figure)

    # Clear existing route traces (assuming 'base' routes are the only Scattermap traces that should be cleared)
    # This loop needs to be careful not to remove the boundary or other fixed elements.
    # A more robust solution might involve assigning a unique name/group to route traces.
    traces_to_keep = []
    for trace in fig.data:
        # Keep the spot boundary trace and any other non-route traces
        if isinstance(trace, go.Scattermap) and \
            (trace.name == spot.name or \
             trace.name == "Контрольные точки" or \
             trace.name == "Старт" or trace.name == "Финиш"):
                traces_to_keep.append(trace)
    
    fig.data = traces_to_keep # Reset data to only include the traces we want to keep
    
    lons = [p.lon for p in processed_route.route.points]
    lats = [p.lat for p in processed_route.route.points]    
    
    center_on_feature(fig, [np.min(lons), np.min(lats), np.max(lons), np.max(lats)])
    add_full_route_to_figure(fig, processed_route)
    
    print_step("Map Drawing", f"Карта обновлена для выбранного маршрута: {processed_route.route.name}")
    return fig