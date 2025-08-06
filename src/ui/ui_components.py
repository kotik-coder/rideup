from typing import List
import dash_bootstrap_components as dbc
from dash import html, dcc
import numpy as np
import plotly.graph_objects as go

from src.routes.trail_features import ElevationSegment, TrailFeatureType
from src.routes.profile_analyzer import SegmentProfile
from src.routes.route import Route
from src.routes.route_processor import ProcessedRoute
from src.ui.trail_style import get_arrow_size, get_feature_color, get_feature_description, get_feature_name, get_gradient_direction, get_segment_name

def create_route_info_card(route: Route, processed_route: ProcessedRoute, segment_profile: SegmentProfile = None):
    """Optimized card using ElevationSegment directly"""
    segments = segment_profile.segments
    
    # Create visualization with direct segment objects
    route_viz = create_segment_visualization(segments)
    
    # Convert long distances to km
    def format_distance(meters):
        if meters >= 1000:
            return f"{meters/1000:.1f} km"
        return f"{meters:.0f} m"
    
    total_length = processed_route.smooth_points[-1].distance_from_origin
    elevs     = [p.elevation for p in processed_route.smooth_points]
    mean_elev = np.mean(elevs)
    elevation_gain = max(elevs) - mean_elev
    elevation_loss = mean_elev - min(elevs)
    
    # Create feature statistics with formatted distances
    feature_stats = {}
    for seg in segment_profile.segments:
        if seg.feature_type:
            name = get_feature_name(seg.feature_type)
            if name not in feature_stats:
                feature_stats[name] = {
                    'count': 0,
                    'total_length': 0,
                    'max_gradient': -float('inf'),
                    'color': get_feature_color(seg),
                    'description': get_feature_description(seg.feature_type)
                }
            feature_stats[name]['count'] += 1
            feature_stats[name]['total_length'] += seg.length()
            feature_stats[name]['avg_gradient'] = seg.avg_gradient() * 100

    return dbc.Card([
        dbc.CardHeader(route.name, className="fw-bold"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div("Total Distance", className="small text-muted"),
                    html.Div(format_distance(total_length), className="h5 mb-0")
                ], width=4),
                dbc.Col([
                    html.Div("Elevation Gain", className="small text-muted"),
                    html.Div(format_distance(elevation_gain), className="h5 mb-0")
                ], width=4),
                dbc.Col([
                    html.Div("Elevation Loss", className="small text-muted"),
                    html.Div(format_distance(elevation_loss), className="h5 mb-0")
                ], width=4),
            ], className="mb-3"),
            
            html.H5("Route Profile", className="mt-3 mb-2 fs-6"),
            html.Div(route_viz, className="mb-3"),
            
            html.H5("Feature Details", className="mt-3 mb-2 fs-6"),
            html.Div(
                _create_feature_details(feature_stats, format_distance),
                className="small"
            )
        ], className="py-2")
    ], style={"maxHeight": "500px", "overflowY": "auto"})

def create_segment_visualization(segments: List[ElevationSegment]):
    """Create keyboard-style visualization with external tooltips"""
    fig = go.Figure()
    
    for i, seg in enumerate(segments):
        color = get_feature_color(seg)
        arrow_size = get_arrow_size(seg)
        direction = get_gradient_direction(seg.avg_gradient())
                        
        # Main segment bar with ID for tooltip targeting
        fig.add_trace(go.Bar(
            x=[i],
            y=[1],
            width=[0.9],
            marker=dict(
                color=color if color else 'white',
                line=dict(width=1, color='#333')
            ),
            hoverinfo='skip',
            showlegend=False,
            customdata=[i]  # Store segment index
        ))
        
        # Direction arrow annotation
        fig.add_annotation(
            x=i,
            y=0.5,
            text=direction,
            showarrow=False,
            font=dict(
                size=arrow_size,
                color='#333' if color == 'white' else 'black'
            ),
            yanchor='middle'
        )
        
        # Short features indicators (red circles below the bar)
        if seg.short_features:
            num_features = len(seg.short_features)
            spacing = 0.8 / max(num_features, 1)
            
            for j, feature in enumerate(seg.short_features):
                fig.add_annotation(
                    x=i - 0.4 + (j + 0.5) * spacing,
                    y=-0.2,
                    text="•",
                    showarrow=False,
                    font=dict(
                        size=14,
                        color='red'
                    ),
                    yanchor='middle'
                )
    
    fig.update_layout(
        margin={'l': 0, 'r': 0, 't': 0, 'b': 20},
        height=60,
        xaxis={'visible': False, 'range': [-0.5, len(segments)-0.5]},
        yaxis={'visible': False, 'range': [-0.5, 1]},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        bargap=0.1
    )
    
    # Create the graph component
    graph = dcc.Graph(
        id='route-profile-graph',
        figure=fig,
        config={'staticPlot': True},
        style={
            'height': '60px',
            'width': '100%',
            'cursor': 'pointer'  # Show pointer cursor to indicate interactivity
        }
    )
    
    # Create invisible divs for each segment that will trigger tooltips
    trigger_divs = [
        html.Div(
            id=f'segment-{i}',
            style={
                'position': 'absolute',
                'left': f'{i*(100/len(segments))}%',
                'width': f'{90/len(segments)}%',
                'height': '60px',
                'zIndex': '100',
                'pointerEvents': 'auto'
            }
        ) for i in range(len(segments))
    ]    
    
    # Wrap everything in a container with relative positioning
    return html.Div(
        [
            html.Div(
                trigger_divs + [graph],
                style={'position': 'relative'}
            )
        ],
        style={'position': 'relative'}
    )

def _create_feature_details(feature_stats: dict, format_distance):
    """Create detailed feature statistics with formatted distances"""
    if not feature_stats:
        return html.P("No trail features detected", className="text-muted")
    
    items = []
    for name, stats in sorted(feature_stats.items(), 
                             key=lambda x: (-x[1]['total_length'], x[0])):
        # Main feature stats
        feature_item = [
            html.Span("■", style={
                "color": stats['color'],
                "fontSize": "1.2em",
                "verticalAlign": "middle"
            }),
            html.Span(f" {name}: ", className="ms-1 fw-bold"),
            html.Span(f"{stats['count']} segments", className="me-2"),
            html.Span(f"{format_distance(stats['total_length'])} total", className="me-2"),
            html.Span(f"avg {stats['avg_gradient']:.1f}%", className="me-2"),
            html.Br(),
            html.Span(stats['description'], className="text-muted font-italic")
        ]
        
        items.append(html.Li(feature_item, className="mb-3"))
    
    return html.Ul(items, className="list-unstyled")

def create_checkpoint_card(checkpoint, route : ProcessedRoute):
    """
    Generates a Dash Bootstrap Components card for displaying checkpoint information.
    Now accepts a Checkpoint object instead of a dictionary.
    """
    
    optional_block = dbc.Col(width=0)
    
    if checkpoint.photo_html:
        optional_block = dbc.Col([
                html.Iframe(
                    srcDoc=checkpoint.photo_html,
                    style={'width': '100%', 'height': '220px', 'border': 'none', 'marginBottom': '5px'}
                )
            ], width=6)
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H5(f"{checkpoint.name}",
                       className="card-title mb-1 fs-6"),
                html.P([html.B("Координаты: "), f"{checkpoint.point.lat:.5f}, {checkpoint.point.lon:.5f}"], className="mb-0", style={'fontSize': '0.9em'}),
                html.P([html.B("Высота: "), f"{checkpoint.point.elevation:.1f} м"], className="mb-0", style={'fontSize': '0.9em'}),
                html.P([html.B("Расстояние от старта: "), f"{checkpoint.distance_from_origin:.1f} м"], className="mb-0", style={'fontSize': '0.9em'}),
                html.P(html.Em(checkpoint.description), className="card-text mb-0", style={'fontSize': '0.9em'})
            ], width=6),
            optional_block
        ], className="g-0"),
        dbc.Row([
            dbc.Col(dbc.Button("← Предыдущий", id="prev-checkpoint-button", className="me-2", color="secondary", size="sm"), width={"size": 6, "offset": 0}),
            dbc.Col(dbc.Button("Следующий →",  id="next-checkpoint-button", className="ms-2", color="secondary", size="sm"), width={"size": 6, "offset": 0})
        ], className="mt-2 text-center")
    ])
    
    