# callbacks.py
import dash
from dash import Input, Output, State, no_update, html
import numpy as np
import plotly.graph_objects as go
from map_helpers import print_step
from route_processor import ProcessedRoute, RouteProcessor
from map_visualization import create_base_map, add_route_to_figure # Removed create_base_map as it's not used directly here
from graph_generation import create_elevation_profile_figure, create_velocity_profile_figure
from spot import SpotLoader, Spot
from ui_components import create_checkpoint_card, create_route_info_card
from typing import List

def setup_callbacks(app, spot: Spot, spot_loader: SpotLoader, route_processor: RouteProcessor, processed_routes: List[ProcessedRoute]):
    """
    Registers all Dash callbacks with the provided Dash app instance.
    Updated to work with the new map_visualization.py functions.
    """

    @app.callback(
        Output('map-graph', 'figure'),
        Output('route-selector', 'options'),
        Output('route-selector', 'value'),
        Output('route-general-info', 'children'),
        Output('checkpoint-info', 'children'),
        Input('initial-load-trigger', 'children'),
        prevent_initial_call=False
    )
    def initial_load_and_display_all(trigger):
        print_step("Callback", "Initial load and display triggered.")

        # Default values for outputs
        # This will create a figure with just the spot boundary
        initial_map_figure = create_base_map(spot) 
        
        route_options = []
        selected_route_value = None
        general_info_children = "Нет доступных маршрутов."
        checkpoint_info_children = "Выберите чекпоинт на карте, чтобы увидеть информацию."

        if not processed_routes:
            print_step("Callback", "No processed routes found. Displaying empty state.")
            return (
                initial_map_figure,
                route_options,
                selected_route_value,
                general_info_children,
                checkpoint_info_children
            )

        # --- Populate route selector options ---
        route_options = [{'label': r.route.name, 'value': i} for i, r in enumerate(processed_routes)]

        # --- Generate Map Figure ---
        # Start with the initial map figure (which already has the spot boundary)
        fig = go.Figure(initial_map_figure) 
        '''
        show_available_routes(fig, spot)
        '''
        print_step("Callback", "Initial display complete.")
        return (
            fig,
            route_options,
            selected_route_value,
            general_info_children,
            checkpoint_info_children
        )