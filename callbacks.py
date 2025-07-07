# callbacks.py
from dash import Input, Output
import dash
from map_helpers import print_step
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
        Input('route-selector', 'value'),
        prevent_initial_call=True
    )
    def handle_route_selection(selected_route_index: int):
        """Processes the selected route and stores the processed data."""
        if selected_route_index is None:
            return dash.no_update, dash.no_update
            
        print_step("Callbacks", f"Processing selected route index: {selected_route_index}")
        selected_route  = spot.routes[selected_route_index]
        processed_route = route_processor.process_route(selected_route)

        create_route_info_card(processed_route) # This line can be removed as update_route_info handles it

        return selected_route_index

    @app.callback(
        Output('route-general-info', 'children'),
        Input('route-data-store', 'data'),
        prevent_initial_call=True
    )
    def update_route_info(processed_route_data):
        """Updates the route information card with the selected route's data."""
        if processed_route_data is None:
            return dash.no_update
            
        from ui_components import create_route_info_card
        return create_route_info_card(processed_route_data)
