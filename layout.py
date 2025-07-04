# layout.py
from typing import List
import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.graph_objects as go
from map_visualization import create_base_map
from route_processor import ProcessedRoute
from spot import Spot

def create_initial_figure(spot: Spot) -> go.Figure:
    """Creates the base Plotly map figure centered on the spot with proper styling."""
    # Create empty figure with spot boundaries
    fig = go.Figure()
    
    # Set initial view to center of spot bounds
    min_lon, min_lat, max_lon, max_lat = spot.bounds
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=12  # Default zoom, will be adjusted by create_base_map
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=False,
        clickmode='event+select'
    )
    
    return fig

def setup_layout(spot: Spot, routes: List[ProcessedRoute] = None) -> dbc.Container:
    """
    Creates the main Dash application layout with map and control panels.
    
    Args:
        spot: Spot object containing the geographic area boundaries
        routes: Optional list of ProcessedRoute objects to display initially
        
    Returns:
        A dbc.Container with the complete application layout including:
        - Map visualization area
        - Route selection controls
        - Information panels
        - Elevation and velocity profile charts
    """
    # Layout configuration constants
    BOTTOM_GRAPHS_HEIGHT_VH = 25  # Height for bottom charts in viewport height units
    TOP_ROW_HEIGHT = f'calc(100vh - {BOTTOM_GRAPHS_HEIGHT_VH}vh - 15px)'  # Dynamic height calculation
    
    # Create initial figure - show routes if provided, otherwise just spot boundaries
    initial_figure = (
        create_base_map(spot, routes) if routes 
        else create_initial_figure(spot))
    
    # Main layout structure
    return dbc.Container(
        [
            # Top row - Map and control panels
            dbc.Row(
                [
                    # Map visualization column (70% width)
                    dbc.Col(
                        dcc.Graph(
                            id='map-graph',
                            figure=initial_figure,
                            style={'height': '100%', 'width': '100%'},
                            config={
                                'displayModeBar': True,
                                'scrollZoom': True,
                                'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                            }
                        ),
                        width=7,
                        className="p-0 h-100"
                    ),
                    
                    # Control panels column (30% width)
                    dbc.Col(
                        [
                            # Route selection card
                            dbc.Card(
                                [
                                    dbc.CardHeader("Route Selection", className="font-weight-bold py-2"),
                                    dbc.CardBody(
                                        dcc.Dropdown(
                                            id='route-selector',
                                            options=[],
                                            placeholder="Select a route...",
                                            className="mb-2",
                                            clearable=False,
                                            searchable=True
                                        ),
                                        className="py-2"
                                    )
                                ],
                                className="mb-3"
                            ),
                            
                            # Route information card
                            dbc.Card(
                                [
                                    dbc.CardHeader("Route Details", className="font-weight-bold py-2"),
                                    dbc.CardBody(
                                        html.Div(id='route-general-info'),
                                        className="py-2"
                                    )
                                ],
                                className="mb-3"
                            ),
                            
                            # Checkpoint information card
                            dbc.Card(
                                [
                                    dbc.CardHeader("Checkpoint Information", className="font-weight-bold py-2"),
                                    dbc.CardBody(
                                        html.Div(id='checkpoint-info'),
                                        className="py-2"
                                    )
                                ]
                            )
                        ],
                        width=5,
                        className="px-3 h-100 scrollable-panel"
                    )
                ],
                className="g-0 m-0",
                style={'height': TOP_ROW_HEIGHT}
            ),
            
            # Bottom row - Data visualization charts
            dbc.Row(
                [
                    # Elevation profile chart
                    dbc.Col(
                        dcc.Graph(
                            id='elevation-profile',
                            style={'height': '100%', 'width': '100%'},
                            config={'displayModeBar': False}
                        ),
                        width=6,
                        className="px-1"
                    ),
                    
                    # Velocity profile chart
                    dbc.Col(
                        dcc.Graph(
                            id='velocity-profile',
                            style={'height': '100%', 'width': '100%'},
                            config={'displayModeBar': False}
                        ),
                        width=6,
                        className="px-1"
                    )
                ],
                className="g-0 mt-2",
                style={'height': f'{BOTTOM_GRAPHS_HEIGHT_VH}vh'}
            ),
            
            # Hidden components for state management
            dcc.Store(id='route-data-store'),
            dcc.Store(id='selected-route-index', data=0),
            dcc.Store(id='selected-checkpoint-index', data=0),
            html.Div(id='initial-load-trigger', style={'display': 'none'})
        ],
        fluid=True,
        className="vh-100 p-0 overflow-hidden"
    )