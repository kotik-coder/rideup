from datetime import datetime, timezone
from typing import List
import dash_bootstrap_components as dbc
from dash import html, dcc
import numpy as np
import plotly.graph_objects as go

from src.routes.spot import Spot
from src.routes.profile_analyzer import Profile, ProfilePoint, ProfileSegment
from src.routes.route import Route
from src.routes.route_processor import ProcessedRoute
from src.ui.trail_style import get_arrow_size, get_feature_color, get_feature_description, get_feature_icon, get_feature_name, get_gradient_direction, get_segment_color, get_segment_description

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
        if traction_difference > 0:
            traction_display = [
                html.Span(f"{base_traction:.0%}", className="text-muted"),
                html.Span(" → ", className="mx-1"),
                html.Span(f"{traction:.0%}", 
                            style={'color': '#1f77b4', 'font-weight': 'bold'})
            ]
        else: 
            traction_display = [html.Span(f"{base_traction:.0%}", className="text-muted")]
        
        # Update the weather section in create_spot_info_card function
        weather_section = []
        if spot.weather:
            # Wind description (unchanged)
            wind_speed = spot.weather.wind_speed
            wind_desc = spot.weather.get_wind_description()

            recency, local_time, hours_old = spot.weather.get_recency()

            if hours_old > 1:
                spot.query_weather()
                recency, local_time, hours_old = spot.weather.get_recency()
            
            # Format weather advice with line breaks
            weather_advice = spot.get_weather_advice()
            advice_items = [html.Div(item) for item in weather_advice.split('\n') if item]
            
            # New precipitation classification display
            precip_class = spot.weather.precipitation_classification
            current_rate = spot.weather.precipitation_now
            forecast_max = spot.weather.precipitation_forecast_max
            three_days_total = spot.weather.precipitation_last_3days
            if three_days_total < 0:
                three_days_total = "N/A"
            else:
                three_days_total = f"{three_days_total:.1f}"
            
            # Get current precipitation classification
            current_precip_class = (
                "Violent" if precip_class['current_rate_violent'] else
                "Heavy" if precip_class['current_rate_heavy'] else
                "Moderate" if precip_class['current_rate_moderate'] else
                "Light" if current_rate > 0 else
                ""
            )
            
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
                            f"{spot.weather.condition} {current_precip_class or ''}",
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
                        html.Small("Precipitation", className="text-muted d-block text-center"),
                        dbc.Row([
                            dbc.Col([
                                html.Div("Current", className="small text-muted text-center"),
                                html.Div(f"{current_rate:.1f} mm/h", className="text-center")
                            ], width=4),
                            dbc.Col([
                                html.Div("Max Forecast", className="small text-muted text-center"),
                                html.Div(f"{forecast_max:.1f} mm/h", className="text-center")
                            ], width=4),
                            dbc.Col([
                                html.Div("3-Day Total", className="small text-muted text-center"),
                                html.Div(f"{three_days_total} mm", className="text-center")
                            ], width=4)
                        ], className="g-0")
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
                            className="small",
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
    profile        = route_data['profile']
    segments       = profile.segments
    profile_points = profile.points
    
    # Create visualization with direct segment objects
    route_viz = create_segment_visualization(profile) if segments else None
    
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
    
    # Create feature statistics with formatted distances - FIXED
    feature_stats = {}
    short_feature_stats = {}
    
    for seg in segments:
        # Process long features
        if seg.feature:
            ftype = seg.feature.feature_type
            name = get_feature_name(ftype)
            feature_length = seg.feature.length(seg, profile_points)
            
            if name not in feature_stats:
                feature_stats[name] = {
                    'count': 0,
                    'total_length': 0,
                    'color': get_feature_color(seg.feature),
                    'description': get_feature_description(ftype)
                }
            
            feature_stats[name]['count'] += 1
            feature_stats[name]['total_length'] += feature_length
        
        # Process short features
        for sf in seg.short_features:
            sf_type = sf.feature_type
            sf_name = get_feature_name(sf_type)
            
            if sf_name not in short_feature_stats:
                short_feature_stats[sf_name] = {
                    'count': 0,
                    'color': get_feature_color(sf),
                    'description': get_feature_description(sf_type)
                }
            
            short_feature_stats[sf_name]['count'] += 1
    
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
            
            # Feature details - FIXED to show both long and short features
            html.H5("Trail Features", className="mt-3 mb-2 fs-6"),
            html.Div(
                _create_feature_details(feature_stats, short_feature_stats, total_length),
                className="small"
            )
        ], className="py-2")
    ])

def _create_feature_details(feature_stats: dict, short_feature_stats: dict, total_route_length: float):
    """Create detailed feature statistics showing long features as percentage of route"""
    if not feature_stats and not short_feature_stats:
        return html.P("No trail features detected", className="text-muted")
    
    items = []
    
    # Process long features - show as percentage of total route
    for name, stats in sorted(feature_stats.items(), 
                             key=lambda x: (-x[1]['total_length'], x[0])):
        percentage = (stats['total_length'] / total_route_length) * 100
        feature_item = [
            html.Span("■", style={
                "color": stats['color'],
                "fontSize": "1.2em",
                "verticalAlign": "middle"
            }),
            html.Span(f" {name}: ", className="ms-1 fw-bold"),
            html.Span(f"{stats['count']} segments", className="me-2"),
            html.Span(f"({percentage:.1f}% of route)", className="text-muted me-2"),
            html.Br(),
            html.Span(stats['description'], className="text-muted font-italic")
        ]
        
        items.append(html.Li(feature_item, className="mb-3"))
    
    # Process short features - only show count, no length
    for name, stats in sorted(short_feature_stats.items(), 
                             key=lambda x: (-x[1]['count'], x[0])):
        feature_item = [
            html.Span("▪", style={
                "color": stats['color'],
                "fontSize": "1.2em",
                "verticalAlign": "middle"
            }),
            html.Span(f" {name}: ", className="ms-1 fw-bold"),
            html.Span(f"{stats['count']} features", className="me-2"),
            html.Br(),
            html.Span(stats['description'], className="text-muted font-italic")
        ]
        
        items.append(html.Li(feature_item, className="mb-3"))
    
    return html.Ul(items, className="list-unstyled")

def create_segment_visualization(profile : Profile):
    """Create visually appealing horizontal table that shows full height without scrolling"""
    segments = profile.segments
    profile_points = profile.points
    
    # Colors that match the route card theme
    header_bg = '#e9ecef'  # Light gray similar to card headers
    border_color = '#dee2e6'  # Subtle border color
    text_color = '#495057'  # Dark gray text
    
    # Create compact table header with segment numbers
    header_cells = [html.Th("Segment", style={
        'min-width': '60px', 
        'font-size': '11px',
        'background-color': header_bg,
        'border': f'1px solid {border_color}',
        'padding': '4px 6px',
        'color': text_color,
        'font-weight': '600'
    })]
    
    for i in range(len(segments)):
        header_cells.append(html.Th(f"{i+1}", style={
            'min-width': '35px', 
            'text-align': 'center',
            'font-size': '11px',
            'background-color': header_bg,
            'border': f'1px solid {border_color}',
            'padding': '4px 2px',
            'color': text_color,
            'font-weight': '600'
        }))
    
    # Create compact gradient row with arrows
    gradient_cells = [html.Td("Gradient", style={
        'background-color': header_bg,
        'font-weight': '600',
        'font-size': '11px',
        'border': f'1px solid {border_color}',
        'padding': '6px 6px',
        'color': text_color
    })]
    
    for seg in segments:
        arrow_size = max(14, min(18, get_arrow_size(seg, profile_points)))  # Constrained size
        direction = get_gradient_direction(seg, profile_points)
        gradient_cells.append(html.Td(
            direction,
            style={
                'text-align': 'center',
                'font-size': f'{arrow_size}px',
                'padding': '4px 2px',
                'height': '22px',
                'border': f'1px solid {border_color}',
                'background-color': 'rgba(248, 249, 250, 0.8)',
                'vertical-align': 'middle'
            }
        ))
    
    # Create compact feature row with colored bars only
    feature_cells = [html.Td("Feature", style={
        'background-color': header_bg,
        'font-weight': '600',
        'font-size': '11px',
        'border': f'1px solid {border_color}',
        'padding': '6px 6px',
        'color': text_color
    })]
    
    for seg in segments:
        if seg.feature or seg.short_features:
            color = get_segment_color(seg)
            feature_cells.append(html.Td(
                style={
                    'background-color': color,
                    'height': '20px',
                    'min-width': '35px',
                    'border-radius': '3px',
                    'padding': '0',
                    'border': f'2px solid {color}',
                    'box-shadow': '0 1px 2px rgba(0,0,0,0.1)',
                    'vertical-align': 'middle'
                },
                title=get_segment_description(seg, profile_points)
            ))
        else:
            feature_cells.append(html.Td(
                style={
                    'background-color': 'transparent',
                    'height': '20px',
                    'border': f'1px solid {border_color}',
                    'vertical-align': 'middle'
                }
            ))
    
    # Create compact short features row with icons
    short_feature_cells = [html.Td("Extra", style={
        'background-color': header_bg,
        'font-weight': '600',
        'font-size': '11px',
        'border': f'1px solid {border_color}',
        'padding': '6px 6px',
        'color': text_color
    })]
    
    for seg in segments:
        if seg.short_features:
            icon_elements = []
            for feature in seg.short_features:
                feature_color = get_feature_color(feature)
                icon = get_feature_icon(feature.feature_type)
                icon_elements.append(html.Span(
                    icon,
                    style={
                        'color': feature_color,
                        'font-size': '14px',
                        'margin': '0 1px',
                        'font-weight': 'bold',
                        'text-shadow': '0 1px 1px rgba(0,0,0,0.1)',
                        'display': 'inline-block'
                    },
                    title=get_feature_name(feature.feature_type)
                ))
            short_feature_cells.append(html.Td(
                icon_elements,
                style={
                    'text-align': 'center',
                    'padding': '4px 2px',
                    'height': '22px',
                    'border': f'1px solid {border_color}',
                    'background-color': 'rgba(248, 249, 250, 0.8)',
                    'vertical-align': 'middle'
                }
            ))
        else:
            short_feature_cells.append(html.Td(
                style={
                    'background-color': 'transparent',
                    'height': '22px',
                    'border': f'1px solid {border_color}',
                    'vertical-align': 'middle'
                }
            ))
    
    # Create the compact HTML table
    table = html.Table(
        [
            html.Thead(html.Tr(header_cells)),
            html.Tbody([
                html.Tr(gradient_cells),
                html.Tr(feature_cells),
                html.Tr(short_feature_cells)
            ])
        ],
        style={
            'border-collapse': 'collapse',
            'width': '100%',
            'font-size': '11px',
            'border': f'1px solid {border_color}',
            'border-radius': '6px',
            'overflow': 'hidden',
            'box-shadow': '0 1px 3px rgba(0,0,0,0.05)',
            'background-color': 'white',
            'font-family': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
        }
    )
    
    return html.Div(
        table,
        style={
            'overflowX': 'auto',
            'height': 'auto',  # Allow natural height
            'maxHeight': 'none',  # Remove max height constraint
            'margin-bottom': '12px'
        }
    )

def create_checkpoint_card(checkpoint, route: ProcessedRoute):
    """
    Generates a checkpoint card with selectable list and click-to-enlarge photo.
    """
    # Create list of checkpoint options
    checkpoint_options = [
        {'label': cp.name, 'value': i} 
        for i, cp in enumerate(route.checkpoints)
    ]
    
    current_index = next(
        (i for i, cp in enumerate(route.checkpoints) if cp == checkpoint),
        0
    )
    
    # Create modal for enlarged photo
    photo_modal = dbc.Modal(
        [
            dbc.ModalHeader(
                checkpoint.name,
                close_button=True,
                style={'padding': '0.5rem', 'border-bottom': 'none'}
            ),
            dbc.ModalBody(
                html.Div(
                    style={
                        'height': '75vh',
                        'overflow': 'auto',
                        'padding': '0',
                        'display': 'flex',
                        'justify-content': 'center',
                        'align-items': 'center'
                    },
                    children=html.Iframe(
                        srcDoc=checkpoint.photo_html,
                        style={
                            'width': '100%',
                            'height': '100%',
                            'border': 'none',
                            'min-height': '400px'  # Minimum height guarantee
                        }
                    ) if checkpoint.photo_html else html.Div("No photo available")
                )
            ),
        ],
        id={"type": "checkpoint-photo-modal", "index": current_index},
        size="lg",
        is_open=False,
        style={
            'max-width': '900px',
            'width': '90%',
            'padding': '0',
        },
        contentClassName="p-0",
        backdrop=True
    )
    
    return html.Div([
        photo_modal,
        dbc.Row([
            # Checkpoint list column
            dbc.Col([
                html.H5("Checkpoints", className="card-title mb-2 fs-6"),
                dcc.Dropdown(
                    id='checkpoint-selector',
                    options=checkpoint_options,
                    value=current_index,
                    clearable=False,
                    style={'fontSize': '0.9em'}
                )
            ], width=4),
            
            # Checkpoint details column
            dbc.Col([
                html.H5(checkpoint.name, className="card-title mb-1 fs-6"),
                html.P([html.B("Coordinates: "), f"{checkpoint.point.lat:.5f}, {checkpoint.point.lon:.5f}"], 
                      className="mb-0", style={'fontSize': '0.9em'}),
                html.P([html.B("Elevation: "), f"{checkpoint.point.elevation:.1f} m"], 
                      className="mb-0", style={'fontSize': '0.9em'}),
                html.P([html.B("Distance: "), f"{checkpoint.distance_from_origin:.1f} m"], 
                      className="mb-0", style={'fontSize': '0.9em'}),
                html.P(html.Em(checkpoint.description), 
                      className="card-text mt-2", style={'fontSize': '0.9em'})
            ], width=5),
            
            # Photo column with click handler
            dbc.Col([
                html.Div(
                    html.Div(  # Wrapper div for click handling
                        html.Iframe(
                            srcDoc=checkpoint.photo_html,
                            style={
                                'width': '100%',
                                'height': '220px',
                                'border': 'none',
                                'pointer-events': 'none'  # Allow clicks to pass through
                            }
                        ) if checkpoint.photo_html else None,
                        style={
                            'cursor': 'pointer' if checkpoint.photo_html else 'default',
                            'position': 'relative'
                        },
                        id={"type": "checkpoint-photo-thumbnail", "index": current_index}
                    ),
                    style={'display': 'block' if checkpoint.photo_html else 'none'}
                )
            ], width=3)
        ], className="g-2")
    ])