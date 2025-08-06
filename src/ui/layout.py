import dash_bootstrap_components as dbc
from dash import dcc, html

from src.ui.map_visualization import create_base_map
from src.routes.spot import Spot

def setup_layout(spot: Spot):
    """
    Sets up the main Dash application layout with full-screen map and semi-transparent UI panels.
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
    graph_panel_height = '25vh'
    right_panel_bottom_margin = '30vh'  # Extra space to prevent overlap

    layout = dbc.Container([
        # Full-screen map
        dcc.Graph(
            id='map-graph',
            style={
                'position': 'fixed',
                'top': '0',
                'left': '0',
                'width': '100vw',
                'height': '100vh',
                'z-index': '0'
            },
            figure=map_figure
        ),
        
        # Semi-transparent panels
        html.Div([
            # Right-side panel (route selection and info)
            html.Div([
                dbc.Card([
                    dbc.CardHeader("Маршрут", className="font-weight-bold py-2"),
                    dbc.CardBody([
                        # Route selector
                        dcc.Dropdown(
                            id='route-selector',
                            options=[],
                            placeholder="Выберите маршрут",
                            className="mb-3"
                        ),
                        
                        # Route information
                        html.Div(id='route-general-info', className="mb-3"),
                        
                        # Checkpoint information
                        html.Div(id='checkpoint-info')
                    ], className="py-2")
                ], style={
                    'background-color': 'rgba(255, 255, 255, 0.85)',
                    'height': f'calc(100vh - 40px - {graph_panel_height})',
                    'overflow-y': 'auto'
                })
            ], style={
                'position': 'fixed',
                'top': '20px',
                'right': '20px',
                'width': right_panel_width,
                'max-width': right_panel_max_width,
                'z-index': '1',
                'bottom': right_panel_bottom_margin
            }),
            
            # Bottom panel (single graph with selector)
            html.Div([
                dbc.Row([
                    dbc.Col(
                        dbc.RadioItems(
                            id='graph-selector',
                            options=[
                                {'label': 'Elevation', 'value': 'elevation'},
                                {'label': 'Velocity', 'value': 'velocity'}
                            ],
                            value='elevation',
                            inline=True,
                            className="me-2",
                            style={
                                'padding': '5px',
                                'margin-left': '10px',
                                'font-size': '0.9rem'
                            }
                        ),
                        width=2,
                        style={'padding-right': '0'}
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id='profile-graph',
                            style={'height': '100%', 'width': '100%'}
                        ),
                        width=10,
                        style={'padding-left': '0'}
                    )
                ], style={'height': '100%', 'margin': '0'})
            ], style={
                'position': 'fixed',
                'bottom': '20px',
                'left': '20px',
                'right': f'calc({right_panel_width} + 40px)',
                'height': graph_panel_height,
                'background-color': 'rgba(255, 255, 255, 0.85)',
                'border-radius': '5px',
                'padding': '10px',
                'z-index': '1'
            })
        ]),
        
        # Hidden components
        dcc.Store(id='route-data-store'),
        dcc.Store(id='selected-route-index'),
        dcc.Store(id='selected-checkpoint-index'),  
        dcc.Store(id='map-dimensions-store'),
        dcc.Store(id='initial-state-trigger', data=True),  
        html.Div(id='initial-load-trigger', style={'display': 'none'})
    ], 
    fluid=True, 
    style={
        'padding': '0',
        'margin': '0',
        'height': '100vh',
        'width': '100vw',
        'overflow': 'hidden'
    })

    return layout