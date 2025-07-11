import plotly.graph_objects as go
import numpy as np
from typing import List

from src.routes.statistics_collector import ProfilePoint # For logging, if needed

def get_fill_polygons(distances, values, threshold_value):
    """
    Generates polygon coordinates for areas above and below a given threshold.
    This is a generic function that can be used for both elevation and velocity.

    Args:
        distances (list): List of distances corresponding to data points.
        values (list): List of data values (elevations or velocities).
        threshold_value (float): The threshold value (e.g., median elevation, Q1/Q3 velocity).

    Returns:
        tuple: A tuple containing two lists of polygons:
            - polygons_above (list of lists of tuples): Polygons for areas where value > threshold.
            - polygons_below (list of lists of tuples): Polygons for areas where value < threshold.
    """
    polygons_above = []
    polygons_below = []

    for i in range(len(distances) - 1):
        d1, v1 = distances[i], values[i]
        d2, v2 = distances[i+1], values[i+1]

        is_p1_above = (v1 >= threshold_value)
        is_p2_above = (v2 >= threshold_value)

        if is_p1_above and is_p2_above:
            polygons_above.append([(d1, v1), (d2, v2), (d2, threshold_value), (d1, threshold_value)])
        elif not is_p1_above and not is_p2_above:
            polygons_below.append([(d1, v1), (d2, v2), (d2, threshold_value), (d1, threshold_value)])
        else:
            intersect_d = 0.0
            if v2 - v1 == 0:
                intersect_d = d1 if v1 == threshold_value else (d1 + d2) / 2
            else:
                t = (threshold_value - v1) / (v2 - v1)
                intersect_d = d1 + t * (d2 - d1)

            intersect_point = (intersect_d, threshold_value)

            if is_p1_above and not is_p2_above:
                polygons_above.append([(d1, v1), intersect_point, (d1, threshold_value)])
                polygons_below.append([intersect_point, (d2, v2), (d2, threshold_value), intersect_point])
            elif not is_p1_above and is_p2_above:
                polygons_below.append([(d1, v1), intersect_point, (d1, threshold_value)])
                polygons_above.append([intersect_point, (d2, v2), (d2, threshold_value), intersect_point])
    return polygons_above, polygons_below

def calculate_velocity_quartiles(velocities):
    """Calculates Q1, median (Q2), and Q3 for a list of velocities."""
    if not velocities:
        return 0, 0, 0

    velocities_np = np.array(velocities)
    q1 = np.percentile(velocities_np, 25)
    median = np.percentile(velocities_np, 50)
    q3 = np.percentile(velocities_np, 75)
    return q1, median, q3

# graph_generation.py
import plotly.graph_objects as go
import numpy as np
from statistics_collector import ProfilePoint
from typing import List, Optional

def create_elevation_profile_figure(profile_points: List[ProfilePoint],
                                  highlight_distance: Optional[float] = None) -> go.Figure:
    """Create elevation profile visualization."""
    fig = go.Figure()
    
    if profile_points:
        distances = [p.distance for p in profile_points]
        elevations = [p.elevation for p in profile_points]
        
        fig.add_trace(go.Scatter(
            x=distances,
            y=elevations,
            mode='lines',
            line_color='blue',
            name='Elevation'
        ))

        if highlight_distance is not None:
            idx = np.argmin(np.abs(np.array(distances) - highlight_distance))
            fig.add_trace(go.Scatter(
                x=[distances[idx]],
                y=[elevations[idx]],
                mode='markers',
                marker=dict(size=12, color='gold'),
                name='Highlight'
            ))

    fig.update_layout(
        xaxis_title='Distance (m)',
        yaxis_title='Elevation (m)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=250
    )
    return fig

def create_velocity_profile_figure(profile_points: List[ProfilePoint],
                                 highlight_distance: Optional[float] = None) -> go.Figure:
    """Create velocity profile visualization."""
    fig = go.Figure()
    
    if profile_points:
        distances = [p.distance for p in profile_points]
        velocities = [p.velocity * 3.6 if p.velocity else 0 for p in profile_points]  # Convert m/s to km/h
        
        fig.add_trace(go.Scatter(
            x=distances,
            y=velocities,
            mode='lines',
            line_color='green',
            name='Velocity'
        ))

        if highlight_distance is not None:
            idx = np.argmin(np.abs(np.array(distances) - highlight_distance))
            fig.add_trace(go.Scatter(
                x=[distances[idx]],
                y=[velocities[idx]],
                mode='markers',
                marker=dict(size=12, color='gold'),
                name='Highlight'
            ))

    fig.update_layout(
        xaxis_title='Distance (m)',
        yaxis_title='Velocity (km/h)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=250
    )
    return fig