# callbacks.py
from dash import Input, Output, State, callback_context, no_update
import plotly.graph_objects as go
import dash
from map_helpers import print_step
from map_visualization import update_map_for_selected_route
from route_processor import ProcessedRoute, RouteProcessor
from spot import SpotLoader, Spot
from typing import List

from ui_components import create_checkpoint_card, create_route_info_card

def setup_callbacks(app, spot: Spot, spot_loader: SpotLoader, route_processor: RouteProcessor):
    """
    Registers all Dash callbacks with the provided Dash app instance.
    Handles route selection from both the dropdown and map clicks.
    """

    @app.callback(
        Output('route-selector', 'options'),
        Output('route-selector', 'value'),
        Input('initial-load-trigger', 'children'),
        Input('selected-route-index', 'data'),
        Input('map-graph', 'clickData')
    )
    def sync_dropdown(initial_load, selected_route_index, map_click_data):
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
        State('selected-route-index', 'data'),  # Add current selected index as state
        prevent_initial_call=True
    )
    def update_selected_index(dropdown_value, map_click_data, current_selected_index):
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
                clicked_route_index = point['customdata']
                # Only update if clicking a different route
                if clicked_route_index != current_selected_index:
                    print_step("Callbacks", f"Route {clicked_route_index} selected via map click.")
                    return clicked_route_index
        
        return no_update

    @app.callback(
        Output('route-general-info', 'children'),
        Output('map-graph', 'figure'),
        Output('checkpoint-info', 'children'),
        Output('selected-checkpoint-index', 'data'),
        Input('selected-route-index', 'data'),
        #Input('map-graph', 'clickData'),
        #State('map-graph', 'figure'),
        State('selected-checkpoint-index', 'data'),
        prevent_initial_call=True
    )
    def process_route_and_update_ui(selected_route_index: int, map_click_data, current_map_figure, current_checkpoint_index):
        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]                
        
        if selected_route_index is None:
            return no_update, no_update, no_update, no_update    

        print_step("Callbacks", f"Processing selected route index: {selected_route_index}")
        selected_route = spot.routes[selected_route_index]
        processed_route = route_processor.process_route(selected_route)

        route_info_ui = create_route_info_card(processed_route)
        
        # Pass the current figure and let update_map_for_selected_route handle preserving the view
        updated_map_figure = update_map_for_selected_route(current_map_figure, spot, processed_route, current_checkpoint_index)
        
        # Handle checkpoint selection
        if triggered_id == 'map-graph' and map_click_data:
            point = map_click_data['points'][0]
            if isinstance(point.get('customdata'), list) and len(point.get('customdata', [])) >= 5:
                checkpoint_index = point['customdata'][0]
                checkpoint = processed_route.checkpoints[checkpoint_index + 1]  # +1 to skip start checkpoint
                checkpoint_card = create_checkpoint_card(checkpoint)
                return route_info_ui, updated_map_figure, checkpoint_card, checkpoint_index
        
        # If route changed, reset checkpoint selection
        if triggered_id == 'selected-route-index':
            return route_info_ui, updated_map_figure, None, None
        
        return no_update, no_update, no_update, no_update

    @app.callback(
        Output('selected-checkpoint-index', 'data', allow_duplicate=True),
        Output('checkpoint-info', 'children', allow_duplicate=True),
        Input('prev-checkpoint-button', 'n_clicks'),
        Input('next-checkpoint-button', 'n_clicks'),
        State('selected-route-index', 'data'),
        State('selected-checkpoint-index', 'data'),
        prevent_initial_call=True
    )
    def navigate_checkpoints(prev_clicks, next_clicks, selected_route_index, current_checkpoint_index):
        """
        Handles navigation between checkpoints using the previous/next buttons.
        """
        if selected_route_index is None:
            return no_update, no_update
            
        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        selected_route = spot.routes[selected_route_index]
        processed_route = route_processor.process_route(selected_route)
        
        # Get all checkpoints excluding start and finish
        checkpoints = processed_route.checkpoints
        num_checkpoints = len(checkpoints)
        
        if num_checkpoints == 0:
            return no_update, no_update
            
        if triggered_id == 'prev-checkpoint-button':
            new_index = current_checkpoint_index - 1 if current_checkpoint_index is not None else num_checkpoints - 1
        elif triggered_id == 'next-checkpoint-button':
            new_index = current_checkpoint_index + 1 if current_checkpoint_index is not None else 0
        else:
            return no_update, no_update
            
        # Wrap around if needed
        new_index = new_index % num_checkpoints
        
        checkpoint = checkpoints[new_index]
        checkpoint_card = create_checkpoint_card(checkpoint)
        
        return new_index, checkpoint_card