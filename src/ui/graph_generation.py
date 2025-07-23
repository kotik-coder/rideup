import plotly.graph_objects as go
import numpy as np
from typing import List, Optional
from src.routes.track import TrackAnalysis

from src.routes.statistics_collector import StaticProfilePoint

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

def create_elevation_profile_figure(profile_points: List[StaticProfilePoint],
                                  highlight_distance: Optional[float] = None) -> go.Figure:
    """Create elevation profile visualization with enhanced styling."""
    fig = go.Figure()
    
    distances  = [p.distance_from_origin for p in profile_points]
    elevations = [p.elevation for p in profile_points]
    
    # Calculate mean elevation
    mean_elevation = np.mean(elevations) if elevations else 0
    
    # Get fill polygons for areas above and below mean
    polygons_above, polygons_below = get_fill_polygons(distances, elevations, mean_elevation)
    
    # Add filled areas
    for poly in polygons_above:
        x_vals = [p[0] for p in poly]
        y_vals = [p[1] for p in poly]
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            fill='toself',
            fillcolor='rgba(255, 100, 100, 0.2)',  # Pale red for above mean
            line=dict(width=0),
            hoverinfo='none',
            showlegend=False,
            mode='lines'
        ))
    
    for poly in polygons_below:
        x_vals = [p[0] for p in poly]
        y_vals = [p[1] for p in poly]
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            fill='toself',
            fillcolor='rgba(100, 100, 255, 0.2)',  # Pale blue for below mean
            line=dict(width=0),
            hoverinfo='none',
            showlegend=False,
            mode='lines'
        ))
    
    # Add mean elevation line
    fig.add_hline(
        y=mean_elevation,
        line=dict(
            color='rgba(100, 100, 100, 0.7)',
            width=1,
            dash='dot'
        ),
        annotation_text=f"Mean: {mean_elevation:.1f}m",
        annotation_position="bottom right",
        annotation_font_size=10
    )
    
    # Add shadow effect (wider, semi-transparent line behind main line)
    fig.add_trace(go.Scatter(
        x=distances,
        y=elevations,
        mode='lines',
        line=dict(
            color='rgba(30, 144, 255, 0.3)',
            width=10,
            shape='spline',
            smoothing=1.3
        ),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Main elevation line
    fig.add_trace(go.Scatter(
        x=distances,
        y=elevations,
        mode='lines',
        line=dict(
            color='rgba(0, 100, 255, 1)',
            width=3,
            shape='spline',
            smoothing=1.3
        ),
        hoverinfo='y+name',
        name='Elevation',
        showlegend=False
    ))

    if highlight_distance is not None:
        idx = np.argmin(np.abs(np.array(distances) - highlight_distance))
        fig.add_trace(go.Scatter(
            x=[distances[idx]],
            y=[elevations[idx]],
            mode='markers',
            marker=dict(
                size=12,
                color='gold',
                line=dict(width=2, color='black')
            ),
            hoverinfo='none',
            showlegend=False
        ))

    fig.update_layout(
        xaxis_title='Distance (m)',
        yaxis_title='Elevation (m)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)',
            linecolor='rgba(200, 200, 200, 0.8)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)',
            linecolor='rgba(200, 200, 200, 0.8)'
        )
    )
    return fig

def create_velocity_profile_figure(profile_points: List[TrackAnalysis],
                                 highlight_distance: Optional[float] = None) -> go.Figure:
    """Create velocity profile visualization with quartile-based coloring."""
    fig = go.Figure()
    
    distances = [p.distance_from_start for p in profile_points]
    velocities = [p.horizontal_speed * 3.6 for p in profile_points]  # Convert m/s to km/h
    
    # Calculate velocity quartiles
    q1, median, q3 = calculate_velocity_quartiles(velocities)
    
    # Get fill polygons for different velocity ranges
    _, below_q1_polygons = get_fill_polygons(distances, velocities, q1)
    above_q3_polygons, _ = get_fill_polygons(distances, velocities, q3)
    
    # Add fill for slow sections (bottom quartile)
    for poly in below_q1_polygons:
        x_vals = [p[0] for p in poly]
        y_vals = [p[1] for p in poly]
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            fill='toself',
            fillcolor='rgba(100, 100, 255, 0.2)',  # Pale blue for slow sections
            line=dict(width=0),
            hoverinfo='none',
            showlegend=False,
            mode='lines'
        ))
    
    # Add fill for fast sections (top quartile)
    for poly in above_q3_polygons:
        x_vals = [p[0] for p in poly]
        y_vals = [p[1] for p in poly]
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            fill='toself',
            fillcolor='rgba(255, 100, 100, 0.2)',  # Pale red for fast sections
            line=dict(width=0),
            hoverinfo='none',
            showlegend=False,
            mode='lines'
        ))
    
    # Add quartile reference lines
    fig.add_hline(
        y=q1,
        line=dict(
            color='rgba(100, 100, 255, 0.5)',
            width=1,
            dash='dot'
        ),
        annotation_text=f"Q1: {q1:.1f} km/h",
        annotation_position="bottom right",
        annotation_font_size=10
    )
    
    fig.add_hline(
        y=median,
        line=dict(
            color='rgba(100, 100, 100, 0.7)',
            width=1,
            dash='dash'
        ),
        annotation_text=f"Median: {median:.1f} km/h",
        annotation_position="bottom right",
        annotation_font_size=10
    )
    
    fig.add_hline(
        y=q3,
        line=dict(
            color='rgba(255, 100, 100, 0.5)',
            width=1,
            dash='dot'
        ),
        annotation_text=f"Q3: {q3:.1f} km/h",
        annotation_position="top right",
        annotation_font_size=10
    )
    
    # Add shadow effect
    fig.add_trace(go.Scatter(
        x=distances,
        y=velocities,
        mode='lines',
        line=dict(
            color='rgba(50, 205, 50, 0.3)',
            width=10,
            shape='spline',
            smoothing=1.3
        ),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Main velocity line
    fig.add_trace(go.Scatter(
        x=distances,
        y=velocities,
        mode='lines',
        line=dict(
            color='rgba(0, 180, 0, 1)',
            width=3,
            shape='spline',
            smoothing=1.3
        ),
        hoverinfo='y+name',
        name='Velocity',
        showlegend=False
    ))

    if highlight_distance is not None:
        idx = np.argmin(np.abs(np.array(distances) - highlight_distance))
        fig.add_trace(go.Scatter(
            x=[distances[idx]],
            y=[velocities[idx]],
            mode='markers',
            marker=dict(
                size=12,
                color='gold',
                line=dict(width=2, color='black')
            ),
            hoverinfo='none',
            showlegend=False
        ))

    fig.update_layout(
        xaxis_title='Distance (m)',
        yaxis_title='Velocity (km/h)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)',
            linecolor='rgba(200, 200, 200, 0.8)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)',
            linecolor='rgba(200, 200, 200, 0.8)'
        )
    )
    return fig