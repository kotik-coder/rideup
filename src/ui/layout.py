import dash_bootstrap_components as dbc
from dash import dcc, html
from src.ui.map_visualization import create_base_map
from src.routes.spot import Spot

def setup_layout(spot: Spot):
    """
    Sets up the main Dash application layout with full-screen map and UI panels.
    """
    # Create the base map with tooltip customizations
    map_figure = create_base_map(spot)
    map_figure.update_layout(
        hoverlabel=dict(
            namelength=150,  # Prevents truncation
        )
    )

    # Panel dimensions and positioning
    right_panel_width = '35%'
    right_panel_max_width = '600px'
    vertical_margin = '20px'  # Equal margin for top and bottom
    graph_panel_height = '25vh'

    layout = dbc.Container(
        [
            # Full-screen map
            dcc.Graph(
                id='map-graph',
                figure=map_figure,
                style={
                    'position': 'fixed',
                    'top': '0',
                    'left': '0',
                    'width': '100vw',
                    'height': '100vh',
                    'z-index': '0'
                }
            ),
            
            # Right panel toggle button
            html.Div(
                dbc.Button(
                    html.I(className="fas fa-chevron-left"),
                    id="right-panel-toggle",
                    className="position-absolute",
                    style={
                        'right': '20px',
                        'top': vertical_margin,
                        'z-index': '3',
                        'width': '30px',
                        'height': '30px',
                        'padding': '0',
                        'border-radius': '50%',
                        'background-color': 'white',
                        'transform': 'translateX(100%)'
                    }
                ),
                style={
                    'position': 'fixed',
                    'right': '0',
                    'top': vertical_margin,
                    'z-index': '3'
                }
            ),
            
            # Right-side panel (spot info and route selection)
            html.Div(
                [
                    # Spot info card
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                [
                                    html.Span("Spot Information", style={'flex-grow': '1'}),
                                    dbc.Button(
                                        html.I(className="fas fa-chevron-up"),
                                        id="spot-info-toggle",
                                        className="float-end",
                                        style={
                                            'padding': '0 5px',
                                            'background': 'none',
                                            'border': 'none',
                                            'color': '#6c757d'
                                        }
                                    )
                                ],
                                className="font-weight-bold py-2 d-flex align-items-center"
                            ),
                            dbc.CardBody(
                                html.Div(id='spot-general-info'),
                                className="py-2",
                                id="spot-info-body"
                            )
                        ],
                        id="spot-info-card",
                        style={
                            'background-color': 'rgba(255, 255, 255, 0.85)',
                            'margin-bottom': '10px',
                            'height': '50%',  # Initial height (will be adjusted by callback)
                            'overflow-y': 'auto'
                        }
                    ),
                    
                    # Route selection and info card
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                [
                                    html.Span("Route Information", style={'flex-grow': '1'}),
                                    dbc.Button(
                                        html.I(className="fas fa-chevron-up"),
                                        id="route-info-toggle",
                                        className="float-end",
                                        style={
                                            'padding': '0 5px',
                                            'background': 'none',
                                            'border': 'none',
                                            'color': '#6c757d'
                                        }
                                    )
                                ],
                                className="font-weight-bold py-2 d-flex align-items-center"
                            ),
                            dbc.CardBody(
                                [
                                    dcc.Dropdown(
                                        id='route-selector',
                                        options=[],
                                        placeholder="Select a route",
                                        className="mb-3"
                                    ),
                                    html.Div(id='route-general-info', className="mb-3"),
                                    html.Div(id='checkpoint-info')
                                ],
                                className="py-2",
                                id="route-info-body"
                            )
                        ],
                        id="route-info-card",
                        style={
                            'background-color': 'rgba(255, 255, 255, 0.85)',
                            'height': '50%',  # Initial height (will be adjusted by callback)
                            'overflow-y': 'auto'
                        }
                    )
                ],
                id="right-panel",
                style={
                    'position': 'fixed',
                    'top': vertical_margin,
                    'right': '20px',
                    'width': right_panel_width,
                    'max-width': right_panel_max_width,
                    'z-index': '2',
                    'bottom': vertical_margin,  # Equal margin at bottom
                    'transition': 'all 0.3s ease'
                }
            ),
            
            # Bottom panel toggle button
            html.Div(
                dbc.Button(
                    html.I(className="fas fa-chevron-up"),
                    id="bottom-panel-toggle",
                    className="position-absolute",
                    style={
                        'right': '10px',
                        'top': f'calc({vertical_margin} * -2)',  # Position above bottom panel
                        'z-index': '3',
                        'width': '30px',
                        'height': '30px',
                        'padding': '0',
                        'border-radius': '50%',
                        'background-color': 'white'
                    }
                ),
                style={
                    'position': 'fixed',
                    'bottom': vertical_margin,
                    'left': '20px',
                    'right': f'calc({right_panel_width} + 40px)',
                    'z-index': '3'
                }
            ),
                                                
            # Bottom panel (graph and controls)
            html.Div(
                [
                    # Main content container
                    html.Div(
                        [
                            # Radio buttons column (fixed width)
                            html.Div(
                                dbc.RadioItems(
                                    id='graph-selector',
                                    options=[
                                        {'label': 'Elevation', 'value': 'elevation'},
                                        {'label': 'Velocity', 'value': 'velocity'}
                                    ],
                                    value='elevation',
                                    inline=False,
                                    style={
                                        'padding': '8px',
                                        'background-color': 'rgba(255, 255, 255, 0.85)', 
                                        'border-radius': '8px 0 0 8px',
                                    }
                                ),
                                style={
                                    'width': '110px',  # Slightly reduced width
                                    'height': '100%',
                                    'float': 'left'
                                }
                            ),
                            
                            # Graph container
                            html.Div(
                                dcc.Graph(
                                    id='profile-graph',
                                    config={
                                        'responsive': True,  # Changed from False to True
                                        'autosizable': True,  # Changed from False to True
                                        'displayModeBar': False,
                                        'displaylogo': False
                                    },
                                    style={
                                        'height': '100%',  # Changed to 100% to fill container
                                        'width': '100%',
                                        'margin': '0',
                                        'padding': '0'
                                    }
                                ),
                                style={
                                    'width': 'calc(100% - 110px)',
                                    'height': '100%',
                                    'padding': '8px',
                                    'box-sizing': 'border-box',
                                    'overflow': 'hidden'
                                }
                            )
                        ],
                        style={
                            'height': '100%',
                            'width': '100%',
                            'margin': '0',
                            'padding': '0'
                        }
                    )
                ],
                id='bottom-panel-container',
                style={
                    'position': 'fixed',
                    'bottom': vertical_margin,
                    'left': '20px',
                    'right': f'calc({right_panel_width} + 40px)',
                    'min-height': '180px',  # Slightly reduced minimum
                    'max-height': '280px',  # Slightly reduced maximum
                    'background-color': 'rgba(255, 255, 255, 0.85)', 
                    'border-radius': '8px',
                    'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
                    'z-index': '2',
                    'overflow': 'hidden',
                    'transition': 'all 0.3s ease',                    
                    'height': '0',  # Start minimized
                    'min-height': '0',
                }
            ),
            
            # Hidden components (unchanged)
            dcc.Store(id='route-data-store'),
            dcc.Store(id='selected-route-index'),
            dcc.Store(id='selected-checkpoint-index'),  
            dcc.Store(id='map-dimensions-store'),
            dcc.Store(id='initial-state-trigger', data=True),  
            html.Div(id='initial-load-trigger', style={'display': 'none'}),
            
            # Store components for panel states
            dcc.Store(id='right-panel-state', data={'visible': True}),
            dcc.Store(id='spot-info-state', data={'visible': True}),
            dcc.Store(id='route-info-state', data={'visible': True}),
            dcc.Store(id='bottom-panel-state', data={'visible': False})
        ],
        fluid=True,
        style={
            'padding': '0',
            'margin': '0',
            'height': '100vh',
            'width': '100vw',
            'overflow': 'hidden'
        }
    )

    return layout