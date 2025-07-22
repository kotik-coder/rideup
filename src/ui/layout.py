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
                    dbc.CardHeader("Выбор маршрута", className="font-weight-bold py-2"),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id='route-selector',
                            options=[],
                            placeholder="Выберите маршрут",
                            className="mb-2"
                        )
                    ], className="py-2")
                ], className="mb-2", style={'background-color': 'rgba(255, 255, 255, 0.85)'}),
                
                dbc.Card([
                    dbc.CardHeader("Информация о маршруте", className="font-weight-bold py-2"),
                    dbc.CardBody([
                        html.Div(id='route-general-info')
                    ], className="py-2")
                ], className="mb-2", style={'background-color': 'rgba(255, 255, 255, 0.85)'}),
                
                dbc.Card([
                    dbc.CardHeader("Информация о чекпоинте", className="font-weight-bold py-2"),
                    dbc.CardBody(id='checkpoint-info', className="py-2")
                ], style={'background-color': 'rgba(255, 255, 255, 0.85)'})
            ], style={
                'position': 'fixed',
                'top': '20px',
                'right': '20px',
                'width': '30%',
                'max-width': '400px',
                'z-index': '1',
                'overflow-y': 'auto',
                'max-height': 'calc(100vh - 40px)'
            }),
            
            # Bottom panel (graphs)
            html.Div([
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(
                            id='elevation-profile', 
                            style={'height': '100%', 'width': '100%'}
                        ),
                        width=6,
                        style={'padding': '0 5px'}
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id='velocity-profile', 
                            style={'height': '100%', 'width': '100%'}
                        ),
                        width=6,
                        style={'padding': '0 5px'}
                    )
                ], style={'margin': '0'})
            ], style={
                'position': 'fixed',
                'bottom': '20px',
                'left': '20px',
                'right': '20px',
                'height': '25vh',
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