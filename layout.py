# layout.py
import dash_bootstrap_components as dbc
from dash import dcc, html
from spot import Spot # Import Spot for type hinting
from typing import List
from route import Route # Import Route for type hinting

def setup_layout(spot: Spot) -> dbc.Container:
    """
    Creates the full UI layout for the application, including map, route selector,
    general info, elevation/velocity profiles, and checkpoint details.
    """
    # Initial options for the route selector
    routes = spot.routes
    route_options = [{'label': route.name, 'value': i} for i, route in enumerate(routes)] if routes else []

    return dbc.Container(
        [
            # Hidden component to trigger initial load of map and data
            html.Div(id='initial-load-trigger', style={'display': 'none'}),
            html.Div(id='initial-map-load-trigger', style={'display': 'none'}),


            dbc.Row(
                [
                    # Left Column: Map
                    dbc.Col(
                        dcc.Graph(
                            id='map-graph',
                            style={'height': '100vh', 'width': '100%'},
                            config={
                                'displayModeBar': True,
                                'scrollZoom': True
                            }
                        ),
                        className="p-0",
                        width=8 # Adjust width for map
                    ),
                    # Right Column: Controls and Info
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Выберите маршрут"),
                                    dbc.CardBody(
                                        dcc.Dropdown(
                                            id='route-selector',
                                            options=route_options,
                                            value=None,
                                            clearable=False,
                                            placeholder="Выберите маршрут"
                                        )
                                    )
                                ],
                                className="mb-3"
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader("Общая информация о маршруте"),
                                    dbc.CardBody(id='route-general-info')
                                ],
                                className="mb-3"
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader("Профиль высот"),
                                    dbc.CardBody(dcc.Graph(id='elevation-profile', config={'displayModeBar': False}))
                                ],
                                className="mb-3"
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader("Профиль скорости"),
                                    dbc.CardBody(dcc.Graph(id='velocity-profile', config={'displayModeBar': False}))
                                ],
                                className="mb-3"
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader("Информация о чекпоинте"),
                                    dbc.CardBody(id='checkpoint-info')
                                ],
                                className="mb-3"
                            ),
                            # Hidden dcc.Store components for state management
                            dcc.Store(id='selected-route-index', data=None),
                            dcc.Store(id='selected-checkpoint-index', data=None),
                        ],
                        width=4, # Adjust width for info/controls
                        className="p-3" # Add some padding
                    )
                ],
                className="g-0 m-0" # Remove gutter and margin for rows
            ),
        ],
        fluid=True,
        className="vh-100 p-0" # Make container fluid and take full viewport height
    )
