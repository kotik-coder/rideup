import dash_bootstrap_components as dbc
from dash import dcc, html

from src.ui.map_visualization import create_base_map
from src.routes.spot import Spot

def setup_layout(spot: Spot):
    """
    Sets up the main Dash application layout with enhanced tooltip styling.
    """
    bottom_graphs_vh = 25
    top_row_dynamic_height = f'calc(100vh - {bottom_graphs_vh}vh - 15px)'

    # Create the base map with tooltip customizations
    map_figure = create_base_map(spot)
    map_figure.update_layout(
        hoverlabel=dict(
            namelength=150,  # Prevents truncation
        )
    )

    layout = dbc.Container([
        dbc.Row([
            dbc.Col(dcc.Graph(
                id='map-graph',
                style={'height': '100%', 'width': '100%'},
                figure=map_figure  # Use the pre-configured figure
            ), width=7, style={'padding': '0', 'height': '100%'}),
            dbc.Col([
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
                ], className="mb-2"),
                dbc.Card([
                    dbc.CardHeader("Информация о маршруте", className="font-weight-bold py-2"),
                    dbc.CardBody([
                        html.Div(id='route-general-info')
                    ], className="py-2")
                ], className="mb-2"),
                dbc.Card([
                    dbc.CardHeader("Информация о чекпоинте", className="font-weight-bold py-2"),
                    dbc.CardBody(id='checkpoint-info', className="py-2")
                ])
            ], width=5, style={'padding': '0 15px', 'height': '100%', 'overflowY': 'auto'})
        ], style={'margin': '0', 'height': top_row_dynamic_height}),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='elevation-profile', style={'height': '100%', 'width': '100%'}),
            ], width=6),
            dbc.Col([
                dcc.Graph(id='velocity-profile', style={'height': '100%', 'width': '100%'})
            ], width=6)
        ], style={'margin': '0', 'marginTop': '15px', 'height': f'{bottom_graphs_vh}vh'}),
        dcc.Store(id='route-data-store'),
        dcc.Store(id='selected-route-index'),
        dcc.Store(id='selected-checkpoint-index'),  
        dcc.Store(id='map-dimensions-store'),
        html.Div(id='initial-load-trigger', style={'display': 'none'})
    ], fluid=True, style={'height': '100vh', 'padding': '0', 'overflowY': 'hidden', 'overflowX': 'hidden'})

    return layout