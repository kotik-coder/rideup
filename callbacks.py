# callbacks.py
from dash import Input, Output, State, callback_context, no_update
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
    Handles route selection from both the dropdown and map clicks.
    """

    @app.callback(
        Output('route-selector', 'options'),
        Output('route-selector', 'value'),
        Input('initial-load-trigger', 'children'),
        Input('selected-route-index', 'data')
    )
    def sync_dropdown(initial_load, selected_route_index):
        """
        Populates the dropdown on initial load and syncs its displayed
        value whenever the selected route index changes.
        """
        options = [{'label': route.name, 'value': i} for i, route in enumerate(spot.routes)]
        if not spot.routes:
            return [], None
        
        # Set the dropdown value to the currently selected index
        return options, selected_route_index if selected_route_index is not None else None

    @app.callback(
        Output('selected-route-index', 'data'),
        Input('route-selector', 'value'),
        Input('map-graph', 'clickData'),
        prevent_initial_call=True
    )
    def update_selected_index(dropdown_value, map_click_data):
        """
        Listens to user interactions from the dropdown and the map to update
        the central selected-route-index store.
        """
        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

        if triggered_id == 'route-selector' and dropdown_value is not None:
            print_step("Callbacks", f"Route {dropdown_value} selected via dropdown.")
            return dropdown_value
        
        if triggered_id == 'map-graph' and map_click_data:
            point = map_click_data['points'][0]
            # Base routes on the map have an integer index as customdata.
            if isinstance(point.get('customdata'), int):
                route_index = point['customdata']
                print_step("Callbacks", f"Route {route_index} selected via map click.")
                return route_index
        
        return no_update

    @app.callback(
        Output('route-general-info', 'children'),
        Output('map-graph', 'figure'),
        Input('selected-route-index', 'data'),
        State('map-graph', 'figure'),
        prevent_initial_call=True
    )
    def process_route_and_update_ui(selected_route_index: int, current_map_figure: dict):
        """
        Triggered when the selected route index changes. It processes the
        route and updates the route information card and the map figure.
        """
        if selected_route_index is None:
            return no_update, no_update

        print_step("Callbacks", f"Processing selected route index: {selected_route_index}")
        selected_route = spot.routes[selected_route_index]
        processed_route = route_processor.process_route(selected_route)

        route_info_ui = create_route_info_card(processed_route)
        updated_map_figure = update_map_for_selected_route(current_map_figure, spot, processed_route)

        return route_info_ui, updated_map_figure