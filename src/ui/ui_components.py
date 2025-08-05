import dash_bootstrap_components as dbc
from dash import html
import numpy as np

from src.routes.profile_analyzer import SegmentProfile
from src.routes.route import Route
from src.routes.checkpoints import Checkpoint, PhotoCheckpoint
from src.routes.route_processor import ProcessedRoute

def create_route_info_card(route: Route, processed_route: ProcessedRoute, segment_profile: SegmentProfile = None):
    """
    Generates an enhanced route information card with detailed feature breakdown.
    Args:
        route: The Route object containing basic route information
        processed_route: The ProcessedRoute object containing processed data
        segment_profile: Optional SegmentProfile containing feature analysis
    """
    # Feature descriptions
    FEATURE_DESCRIPTIONS = {
        'Roller': "Repeated undulations (10-50m wavelength)",
        'Switchback': "Sharp 180° turns changing direction",
        'Technical Descent': "Challenging downhill with obstacles",
        'Technical Ascent': "Challenging uphill requiring skill",
        'Flow Descent': "Smooth, rhythmic downhill section",
        'Drop Section': "Sudden steep descent or drop",
        'Short Ascent': "Brief but intense climb (15-50m)",
        'Short Descent': "Brief but steep downhill (15-50m)",
        'Step Up': "Very short, steep climb (<15m)",
        'Step Down': "Very short, steep drop (<15m)",
        'Ascent': "Moderate uphill (1-10% grade)",
        'Descent': "Moderate downhill (-1% to -10% grade)",
        'Steep Ascent': "Very steep climb (>10% grade)",
        'Steep Descent': "Very steep downhill (<-10% grade)"
    }

    # Basic route stats
    mean_elevation = np.mean([p.elevation for p in route.points])
    max_elevation = max(p.elevation for p in route.points)
    min_elevation = min(p.elevation for p in route.points)
    elevation_gain = max_elevation - min_elevation
    
    # Feature statistics
    feature_stats = {}
    
    if segment_profile and hasattr(segment_profile, 'segments'):
        segments = segment_profile.segments
    elif hasattr(processed_route, 'segments'):
        segments = processed_route.segments
    else:
        segments = []
    
    for segment in segments:
        feature_type = segment.feature_type if segment.feature_type else segment.gradient_type
        if feature_type:
            feature_name = feature_type.name.replace('_', ' ').title()
            if feature_name not in feature_stats:
                feature_stats[feature_name] = {
                    'count': 0,
                    'total_length': 0,
                    'avg_gradient': [],
                    'description': FEATURE_DESCRIPTIONS.get(feature_name, "Trail feature")
                }
            feature_stats[feature_name]['count'] += 1
            feature_stats[feature_name]['total_length'] += segment.length()
            feature_stats[feature_name]['avg_gradient'].append(segment.avg_gradient())
    
    # Create feature list items with tooltips
    feature_items = []
    for feature, stats in feature_stats.items():
        avg_grad = np.mean(stats['avg_gradient']) * 100
        feature_items.append(
            dbc.ListGroupItem([
                html.Div([
                    html.Span(
                        feature,
                        className="fw-bold",
                        id=f"tooltip-{feature.lower().replace(' ', '-')}",
                        style={"cursor": "pointer"}
                    ),
                    dbc.Tooltip(
                        stats['description'],
                        target=f"tooltip-{feature.lower().replace(' ', '-')}",
                        placement="top"
                    ),
                    html.Span(f" ×{stats['count']}", className="text-muted ms-2")
                ]),
                html.Div([
                    html.Span(f"{stats['total_length']:.0f}m", className="me-2"),
                    html.Span(f"avg {avg_grad:.1f}%", className="text-muted")
                ], className="small")
            ])
        )

    # Rest of the card layout remains the same...
    return dbc.Card([
        dbc.CardHeader(route.name, className="fw-bold"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div("Total Distance", className="small text-muted"),
                    html.Div(f"{route.total_distance:.0f}m", className="h5 mb-0")
                ], width=4),
                dbc.Col([
                    html.Div("Elevation Gain", className="small text-muted"),
                    html.Div(f"{elevation_gain:.0f}m", className="h5 mb-0")
                ], width=4),
                dbc.Col([
                    html.Div("Avg Elevation", className="small text-muted"),
                    html.Div(f"{mean_elevation:.0f}m", className="h5 mb-0")
                ], width=4),
            ], className="mb-3"),
            
            html.H5("Trail Features", className="mt-3 mb-2 fs-6"),
            dbc.ListGroup(feature_items, flush=True, className="small"),
            
            html.H5("Technical Sections", className="mt-3 mb-2 fs-6"),
            html.Div(
                _create_technical_summary(feature_stats),
                className="small"
            )
        ], className="py-2")
    ], style={"maxHeight": "400px", "overflowY": "auto"})

def _create_technical_summary(feature_stats):
    """Helper to create technical difficulty summary"""
    tech_features = [
        'Technical Ascent', 'Technical Descent',
        'Drop Section', 'Steep Ascent', 'Steep Descent'
    ]
    
    tech_items = []
    total_tech_length = 0
    
    for feature in tech_features:
        if feature in feature_stats:
            length = feature_stats[feature]['total_length']
            total_tech_length += length
            tech_items.append(
                html.Li(f"{feature}: {length:.0f}m")
            )
    
    if not tech_items:
        return html.P("No significant technical sections", className="text-muted")
    
    tech_percentage = (total_tech_length / sum(
        stats['total_length'] for stats in feature_stats.values()
    )) * 100
    
    return html.Ul([
        *tech_items,
        html.Li(f"Total Technical: {total_tech_length:.0f}m ({tech_percentage:.0f}%)", 
               className="fw-bold mt-1")
    ], className="list-unstyled")

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
    
    