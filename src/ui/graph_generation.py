import plotly.graph_objects as go
import numpy as np
from typing import Any, Dict, List, Optional
from src.routes.baseline import Baseline
from src.routes.profile_analyzer import Profile
from src.routes.track import TrackAnalysis

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

def create_elevation_profile_figure(profile: Profile,
                                    highlight_distance: Optional[float] = None) -> go.Figure:
    """Create elevation profile with baseline, median reference, and gradient visualization."""
    fig = go.Figure()
    
    all_points = [p for p in profile.points]
    distances = [p.distance_from_origin for p in all_points]
    elevations = [p.elevation for p in all_points]
    baselines = [p.baseline for p in all_points]
    gradients = [p.gradient * 100 for p in all_points]  # Convert to percentage
    
    has_baseline = len(baselines) == len(all_points)    
    
    # Calculate median elevation
    median_elevation = np.median(elevations) if elevations else 0
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Main elevation line (primary y-axis)
    fig.add_trace(go.Scatter(
        x=distances,
        y=elevations,
        mode='lines',
        line=dict(
            color='rgba(0, 100, 255, 1)',  # Solid blue
            width=3,
            shape='spline',
            smoothing=1.3
        ),
        name='Elevation',
        yaxis='y1'
    ))

    # Gradient line (secondary y-axis)
    fig.add_trace(go.Scatter(
        x=distances,
        y=gradients,
        mode='lines',
        line=dict(
            color='rgba(255, 100, 0, 0.7)',  # Orange
            width=2,
            shape='spline',
            smoothing=1.3,
            dash='dot'
        ),
        name='Gradient',
        yaxis='y2'
    ))

    if has_baseline:
        # Baseline line - subtle dashed
        fig.add_trace(go.Scatter(
            x=distances,
            y=baselines,
            mode='lines',
            line=dict(
                color='rgba(100, 100, 100, 0.8)',
                width=1.5,
                dash='dash',
                shape='spline',
                smoothing=1.3
            ),
            name='Baseline',
            yaxis='y1'
        ))

        # Unified fill between elevation and baseline
        fig.add_trace(go.Scatter(
            x=np.concatenate([distances, distances[::-1]]),
            y=np.concatenate([elevations, baselines[::-1]]),
            fill='toself',
            fillcolor='rgba(100, 180, 255, 0.3)',  # Light blue
            line=dict(width=0),
            name='Oscillations',
            yaxis='y1'
        ))

    # Median reference line
    fig.add_hline(
        y=median_elevation,
        line=dict(
            color='rgba(100, 100, 100, 0.7)',
            width=1,
            dash='dot'
        ),
        annotation_text=f"Median: {median_elevation:.1f}m",
        annotation_position="bottom right",
        yref='y1'
    )

    # Zero gradient reference line
    fig.add_hline(
        y=0,
        line=dict(
            color='rgba(255, 100, 0, 0.5)',
            width=1,
            dash='dot'
        ),
        annotation_text="0% grade",
        annotation_position="top right",
        yref='y2'
    )

    # Shadow effect under main line
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
        showlegend=False,
        yaxis='y1'
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
            showlegend=False,
            yaxis='y1'
        ))

    fig.update_layout(
        xaxis_title='Distance (m)',
        yaxis=dict(
            title='Elevation (m)',
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)',
            linecolor='rgba(200, 200, 200, 0.8)'
        ),
        yaxis2=dict(
            title='Gradient (%)',
            overlaying='y',
            side='right',
            showgrid=False,
            range=[min(gradients + [-20]), max(gradients + [20])],  # Add buffer
            linecolor='rgba(255, 100, 0, 0.7)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        autosize=True,
        margin=dict(l=20, r=60, t=20, b=20),  # Reduced top margin
    )
    return fig

def create_velocity_profile_figure(actual_profile_points: List[TrackAnalysis],
                                 theoretical_profile_points: Optional[List[TrackAnalysis]] = None,
                                 highlight_distance: Optional[float] = None) -> go.Figure:
    """Create velocity profile visualization with actual and theoretical data"""
    fig = go.Figure()
    
    # Process actual velocity data
    actual_distances = [p.distance_from_start for p in actual_profile_points]
    actual_velocities = [p.horizontal_speed * 3.6 for p in actual_profile_points]  # Convert m/s to km/h
    
    # Add actual velocity line
    fig.add_trace(go.Scatter(
        x=actual_distances,
        y=actual_velocities,
        mode='lines',
        line=dict(
            color='rgba(0, 180, 0, 1)',  # Green for actual data
            width=3,
            shape='spline',
            smoothing=1.3
        ),
        name='Actual Velocity',
        hoverinfo='y+name'
    ))
    
    # Process theoretical velocity data if available
    if theoretical_profile_points:
        theoretical_distances = [p.distance_from_start for p in theoretical_profile_points]
        theoretical_velocities = [p.horizontal_speed * 3.6 for p in theoretical_profile_points]
        
        # Add theoretical velocity line
        fig.add_trace(go.Scatter(
            x=theoretical_distances,
            y=theoretical_velocities,
            mode='lines',
            line=dict(
                color='rgba(255, 100, 0, 1)',  # Orange for theoretical data
                width=3,
                shape='spline',
                smoothing=1.3,
                dash='dash'
            ),
            name='Theoretical Velocity',
            hoverinfo='y+name'
        ))
    
    # Calculate velocity quartiles for actual data
    q1, median, q3 = calculate_velocity_quartiles(actual_velocities)
    
    # Get fill polygons for different velocity ranges (actual data only)
    _, below_q1_polygons = get_fill_polygons(actual_distances, actual_velocities, q1)
    above_q3_polygons, _ = get_fill_polygons(actual_distances, actual_velocities, q3)
    
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
    
    # Add shadow effect for actual data
    fig.add_trace(go.Scatter(
        x=actual_distances,
        y=actual_velocities,
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

    if highlight_distance is not None:
        # Highlight actual data point
        actual_idx = np.argmin(np.abs(np.array(actual_distances) - highlight_distance))
        fig.add_trace(go.Scatter(
            x=[actual_distances[actual_idx]],
            y=[actual_velocities[actual_idx]],
            mode='markers',
            marker=dict(
                size=12,
                color='gold',
                line=dict(width=2, color='black')
            ),
            name='Highlight',
            showlegend=False
        ))
        
        # Highlight theoretical data point if available
        if theoretical_profile_points:
            theoretical_idx = np.argmin(np.abs(np.array(theoretical_distances) - highlight_distance))
            fig.add_trace(go.Scatter(
                x=[theoretical_distances[theoretical_idx]],
                y=[theoretical_velocities[theoretical_idx]],
                mode='markers',
                marker=dict(
                    size=12,
                    color='orange',
                    line=dict(width=2, color='black')
                ),
                name='Theoretical Highlight',
                showlegend=False
            ))

    fig.update_layout(
        xaxis_title='Distance (m)',
        yaxis_title='Velocity (km/h)',
        autosize=True,
        margin=dict(l=20, r=60, t=20, b=20),
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
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

def create_comparison_velocity_figure(velocity_data: Dict[str, Any],
                                      highlight_distance: Optional[float] = None) -> go.Figure:
    """Create a comparison figure showing multiple theoretical profiles alongside actual data"""
    fig = go.Figure()
    
    # Add actual velocity data if available
    if velocity_data.get('actual'):
        actual_distances = [p.distance_from_start for p in velocity_data['actual']]
        actual_velocities = [p.horizontal_speed * 3.6 for p in velocity_data['actual']]
        
        fig.add_trace(go.Scatter(
            x=actual_distances,
            y=actual_velocities,
            mode='lines',
            line=dict(color='rgba(0, 0, 0, 1)', width=4, shape='spline'),
            name='Actual Velocity',
            hoverinfo='y+name'
        ))
    
    # Add theoretical profiles for different abilities
    theoretical_profiles = velocity_data.get('theoretical', {})
    colors = ['rgba(255, 0, 0, 0.8)', 'rgba(255, 165, 0, 0.8)', 'rgba(0, 128, 0, 0.8)', 
              'rgba(0, 0, 255, 0.8)', 'rgba(128, 0, 128, 0.8)']
    
    for i, (ability_name, profile_data) in enumerate(theoretical_profiles.items()):
        if i < len(colors):
            analysis_points = profile_data.get('analysis_points', [])
            if analysis_points:
                distances = [p.distance_from_start for p in analysis_points]
                velocities = [p.horizontal_speed * 3.6 for p in analysis_points]
                
                fig.add_trace(go.Scatter(
                    x=distances,
                    y=velocities,
                    mode='lines',
                    line=dict(color=colors[i], width=2, shape='spline', dash='dot'),
                    name=f'{ability_name} Theoretical',
                    hoverinfo='y+name'
                ))
    
    fig.update_layout(
        xaxis_title='Distance (m)',
        yaxis_title='Velocity (km/h)',
        autosize=True,
        margin=dict(l=20, r=60, t=20, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig