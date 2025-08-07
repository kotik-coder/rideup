from datetime import datetime, timezone
from typing import List
import dash_bootstrap_components as dbc
from dash import html, dcc
import numpy as np
import plotly.graph_objects as go

from src.routes.spot import Spot
from src.routes.trail_features import ElevationSegment
from src.routes.route import Route
from src.routes.route_processor import ProcessedRoute
from src.ui.trail_style import get_arrow_size, get_feature_color, get_feature_description, get_feature_name, get_gradient_direction

def create_spot_info_card(spot: Spot):
    """Compact spot card with enhanced weather display"""
    card_children = [
        dbc.CardHeader(spot.name, className="font-weight-bold py-2")
    ]

    if not spot.terrain:
        card_children.append(
            dbc.CardBody([html.P("No terrain data available", className="text-muted mb-0")])
        )
    else:
        # Prepare terrain data
        total_routes = len(spot.routes)
        total_distance = sum(r.total_distance for r in spot.routes) / 1000
        
        # Calculate adjusted traction if weather data exists
        base_traction = spot.terrain.traction_score
        traction = spot.get_traction()
        
        # Determine if we should highlight the adjusted score
        traction_difference = abs(base_traction - traction)
        show_adjusted = spot.weather and traction_difference > 0.05  # Only show if significant difference
        
        # Surface composition list
        surface_list = []
        if spot.terrain.surface_types:
            surfaces = sorted(spot.terrain.surface_types.items(), key=lambda x: -x[1])[:5]
            surface_list = [
                html.Span([
                    html.Span(spot.terrain.surface_icons.get(surface.lower(), '▪'), className="me-1"),
                    f"{surface.capitalize()} ({percent:.0%})"
                ]) for surface, percent in surfaces
            ]
            for i in range(len(surface_list)-1, 0, -1):
                surface_list.insert(i, html.Span(", ", className="mx-1"))

        # Build traction display
        traction_display = []
        if show_adjusted:
            traction_display.extend([
                html.Span(f"{base_traction:.0%}", className="text-muted"),
                html.Span(" → ", className="mx-1"),
                html.Span(f"{traction:.0%}", 
                         style={'color': '#1f77b4', 'font-weight': 'bold'})
            ])
        else:
            traction_display.append(
                html.Span(f"{base_traction:.0%}")
            )

        # Enhanced weather section
        weather_section = []
        if spot.weather:
            # Wind description
            wind_speed = spot.weather.wind_speed
            wind_desc = (
                "Calm" if wind_speed < 0.5 else
                "Light breeze" if wind_speed < 3.3 else
                "Moderate breeze" if wind_speed < 5.5 else 
                "Strong wind"
            )
            
            last_updated_utc = spot.weather.last_updated
            last_updated_local = last_updated_utc.astimezone()
            
            # Calculate age correctly
            current_utc = datetime.now(timezone.utc)
            hours_old = (current_utc - last_updated_utc).total_seconds() / 3600
            
            recency = ("🟢" if hours_old < 1 else 
                    "🟡" if hours_old < 3 else 
                    "🔴")
            
            local_time = last_updated_local.strftime('%Y-%m-%d %H:%M')
            
            # Format weather advice with line breaks
            weather_advice = spot.get_weather_advice()
            advice_items = [html.Div(item) for item in weather_advice.split('\n') if item]
            
            weather_section = [
                html.Hr(className="my-2"),
                dbc.Row([
                    dbc.Col([
                        html.Small("Current Conditions", className="text-muted d-block"),
                        html.Div([
                            html.Img(
                                src=spot.weather.icon_url,
                                style={'height': '24px', 'margin-right': '8px'}
                            ),
                            spot.weather.condition,
                        ], className="d-flex align-items-center")
                    ], width=6),
                    dbc.Col([
                        html.Small("Temperature", className="text-muted d-block"),
                        f"{spot.weather.temperature:.1f}°C"
                    ], width=6)
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col([
                        html.Small("Wind Speed", className="text-muted d-block"),
                        f"{wind_speed:.1f} m/s ({wind_desc})"
                    ], width=6),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                html.Small("Precipitation (mm)", className="text-muted d-block text-center"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div("3d", className="small text-muted text-center"),
                                        html.Div(f"{spot.weather.precipitation_last_3days:.1f}", className="text-center")
                                    ], width=4),
                                    dbc.Col([
                                        html.Div("Now", className="small text-muted text-center"),
                                        html.Div(f"{spot.weather.precipitation_now:.1f}", className="text-center")
                                    ], width=4),
                                    dbc.Col([
                                        html.Div("Next 8h", className="small text-muted text-center"),
                                        html.Div(f"{sum(f['precip'] for f in spot.weather.hourly_forecast):.1f}", className="text-center")
                                    ], width=4)
                                ], className="g-0")
                            ], width=12)
                        ], className="mb-2")
                    ], width=6)
                ], className="mb-2"),
                html.Div([
                    html.Small("Riding Advice", className="text-muted d-block mb-1"),
                    html.Div(
                        advice_items,
                        className="p-2",
                        style={
                            'background-color': '#f8f9fa',
                            'border-radius': '4px',
                            'font-size': '0.85rem'
                        }
                    )
                ]),
                html.Div([
                    html.Small(
                        f"{recency} Updated: {local_time}",
                        className="text-muted d-block text-end"
                    )
                ], className="mt-1")
            ]

        # Build card body with reorganized primary stats
        card_children.append(
            dbc.CardBody([
                # Primary stats in 3 columns
                dbc.Row([
                    dbc.Col([
                        html.Small("Primary Surface", className="text-muted d-block"),
                        f"{spot.terrain.dominant_surface.capitalize()}"
                    ], width=4),
                    dbc.Col([
                        html.Small("Traction", className="text-muted d-block"), 
                        html.Div(traction_display)
                    ], width=4),
                    dbc.Col([
                        html.Small("Recommended Bike", className="text-muted d-block"),
                        html.Div(
                            spot._recommend_bike_type(
                                spot.terrain.dominant_surface,
                                traction
                            ),
                            className="small"
                        )
                    ], width=4)
                ], className="mb-3"),
                
                # Surface composition
                html.Div([
                    html.Span("Surface Composition:", className="text-muted d-block mb-1"),
                    html.Div(surface_list, className="d-inline")
                ], className="small mb-3"),
                
                # Route stats
                dbc.Row([
                    dbc.Col([
                        html.Small("Total Routes", className="text-muted d-block"),
                        str(total_routes)
                    ], width=6),
                    dbc.Col([
                        html.Small("Total Distance", className="text-muted d-block"),
                        f"{total_distance:.1f} km"
                    ], width=6)
                ]),
                
                # Weather section
                *weather_section
            ], className="py-2")
        )

    return dbc.Card(
        card_children,
        style={
            'background-color': 'rgba(255, 255, 255, 0.95)',
            'font-size': '0.9rem'
        }
    )

def create_route_info_card(route: Route, processed_route: ProcessedRoute, route_data: dict):
    """Optimized card with difficulty rating and enhanced elevation display"""
    segments = route_data['segments'].segments
    
    # Create visualization with direct segment objects
    route_viz = create_segment_visualization(segments) if segments else None
    
    # Convert long distances to km
    def format_distance(meters):
        if meters >= 1000:
            return f"{meters/1000:.1f} km"
        return f"{meters:.0f} m"
    
    total_length = processed_route.smooth_points[-1].distance_from_origin
    elevs = [p.elevation for p in processed_route.smooth_points]
    mean_elev = np.mean(elevs)
    elevation_gain = max(elevs) - mean_elev
    elevation_loss = mean_elev - min(elevs)
    
    # Get difficulty rating
    difficulty = route_data.get('difficulty', 'GREEN')
    difficulty_color = {
        'GREEN': 'success',
        'BLUE': 'primary',
        'BLACK': 'dark',
        'DOUBLE_BLACK': 'danger'
    }.get(difficulty, 'secondary')
    
    # Create feature statistics with formatted distances
    feature_stats = {}
    for seg in segments:
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
        dbc.CardHeader([
            html.Div(route.name, className="fw-bold d-inline"),
            dbc.Badge(difficulty, color=difficulty_color, className="ms-2")
        ], className="d-flex align-items-center"),
        dbc.CardBody([
            # Route statistics section
            dbc.Row([
                dbc.Col([
                    html.Div("Elevation", className="small text-muted"),
                    html.Div([
                        html.Span(f"+{elevation_gain:.0f}m", className="text-success"),
                        html.Span(" / ", className="mx-1"),
                        html.Span(f"-{elevation_loss:.0f}m", className="text-danger")
                    ], className="h5 mb-0")
                ], width=4),
                dbc.Col([
                    html.Div("Distance", className="small text-muted"),
                    html.Div(format_distance(total_length), className="h5 mb-0")
                ], width=4),
                dbc.Col([
                    html.Div("Difficulty", className="small text-muted"),
                    html.Div(difficulty, className="h5 mb-0")
                ], width=4),
            ], className="mb-3"),
            
            # Route profile visualization
            html.H5("Route Profile", className="mt-3 mb-2 fs-6"),
            html.Div(route_viz, className="mb-3") if route_viz else None,
            
            # Feature details
            html.H5("Feature Details", className="mt-3 mb-2 fs-6"),
            html.Div(
                _create_feature_details(feature_stats, format_distance) if feature_stats else 
                html.P("No trail features detected", className="text-muted"),
                className="small"
            )
        ], className="py-2")
    ])

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
    
    