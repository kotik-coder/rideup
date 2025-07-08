# callbacks.py
from dash import Input, Output, State
import plotly.graph_objects as go
import dash
from map_helpers import print_step
from map_visualization import update_map_for_selected_route
from route_processor import ProcessedRoute, RouteProcessor
from spot import SpotLoader, Spot
from typing import List

from ui_components import create_route_info_card

def setup_callbacks(app, spot: Spot, spot_loader: SpotLoader, route_processor: RouteProcessor):
    """
    Registers all Dash callbacks with the provided Dash app instance.
    Updated to work with the new map_visualization.py functions.
    """
    
    @app.callback(
    Output('route-selector', 'options'),
    Output('route-selector', 'value'),
    Input('initial-load-trigger', 'children'),
    Input('selected-route-index', 'data')
    )
    def initialize_dropdown(_, selected_route_index):
        """Populates the dropdown and sets a default value."""
        if not spot.routes:
            return [], None
        
        options = [{'label': route.name, 'value': i} for i, route in enumerate(spot.routes)]
        default_value = None
        if selected_route_index:
            default_value = selected_route_index
        return options, default_value
    
    @app.callback(
        Output('selected-route-index', 'data'),
        Output('route-general-info', 'children'),
        Output('map-graph', 'figure'),
        Input('route-selector', 'value'),
        State('map-graph', 'figure'),
        prevent_initial_call=True
    )
    def handle_route_selection(selected_route_index: int, current_map_figure: dict):
        """Processes the selected route and updates the UI directly."""
        if selected_route_index is None:
            return dash.no_update, dash.no_update, dash.no_update # Added an extra dash.no_update

        print_step("Callbacks", f"Processing selected route index: {selected_route_index}")
        selected_route = spot.routes[selected_route_index]
        processed_route = route_processor.process_route(selected_route)

        # Generate the UI components using create_route_info_card
        route_info_ui = create_route_info_card(processed_route)
        
        # Update the map using the new helper function
        updated_map_figure = update_map_for_selected_route(current_map_figure, spot, processed_route)

        return selected_route_index, route_info_ui, updated_map_figure