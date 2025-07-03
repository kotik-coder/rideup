from typing import List
import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.graph_objects as go
from map_visualization import calculate_zoom, add_forest_boundary_and_name_to_figure

def create_initial_figure(bounds: List[float]) -> go.Figure:
    """Creates the base Plotly map figure with centering and style."""
    min_lon_val, min_lat_val, max_lon_val, max_lat_val = bounds

    center_lat = (min_lat_val + max_lat_val) / 2
    center_lon = (min_lon_val + max_lon_val) / 2

    initial_zoom = calculate_zoom([min_lat_val, max_lat_val], [min_lon_val, max_lon_val])

    fig = go.Figure(go.Scattermap())
    fig.update_layout(
        map_style="open-street-map",
        map=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=initial_zoom
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=False,
        clickmode='event+select'
    )

    add_forest_boundary_and_name_to_figure(fig, bounds)
    return fig

def setup_layout(initial_bounds):
    """
    Sets up the main Dash application layout.
    Requires initial_bounds from RouteManager to create the initial map figure.
    """
    bottom_graphs_vh = 25
    top_row_dynamic_height = f'calc(100vh - {bottom_graphs_vh}vh - 15px)'

    layout = dbc.Container([
        dbc.Row([
            dbc.Col(dcc.Graph(
                id='map-graph',
                style={'height': '100%', 'width': '100%'},
                figure=create_initial_figure(initial_bounds)
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
        html.Div(id='initial-load-trigger', style={'display': 'none'})
    ], fluid=True, style={'height': '100vh', 'padding': '0', 'overflowY': 'hidden', 'overflowX': 'hidden'})

    return layout