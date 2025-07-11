# callbacks.py
from dash import Input, Output, State, callback_context, no_update

from src.ui.map_helpers import print_step
from src.ui.map_visualization import *
from src.routes.route_processor import RouteProcessor
from src.routes.spot import Spot

from src.ui.ui_components import create_checkpoint_card, create_route_info_card

def setup_callbacks(app, spot: Spot, route_processor: RouteProcessor):
    """
    Registers all Dash callbacks with the provided Dash app instance.
    Handles route selection from both the dropdown and map clicks.
    """

    @app.callback(
        Output('route-selector', 'options'),
        Output('route-selector', 'value'),
        Input('initial-load-trigger', 'children'),
        Input('selected-route-index', 'data'),
        State('route-selector', 'value'),
        prevent_initial_call=True
    )
    def sync_dropdown(initial_load, selected_route_index, current_dropdown_value):
        """
        Populates the dropdown on initial load and syncs its displayed
        value whenever the selected route index changes.
        """
        options = [{'label': route.name, 'value': i} for i, route in enumerate(spot.routes)]
        
        if not spot.routes:
            return [], None
        
        # Only update the value if it's different from current
        if selected_route_index != current_dropdown_value:
            return options, selected_route_index            
        
        # Otherwise, just return the options (no value change)
        return options, no_update

    @app.callback(
        Output('selected-route-index', 'data'),            
        Output('route-general-info', 'children'),
        Output('map-graph', 'figure'),        
        Input('route-selector', 'value'),
        Input('map-graph', 'clickData'),
        State('selected-route-index', 'data'),  # Add current selected index as state
        State('map-graph', 'figure'),
        prevent_initial_call=True
    )
    def update_selected_index(dropdown_value, map_click_data, current_selected_index, current_map_figure):
        """
        Listens to user interactions from the dropdown and the map to update
        the central selected-route-index store.
        """
        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        
        result = no_update

        if triggered_id == 'route-selector' and dropdown_value is not None:
            # Only update if the selected value is different from current
            if dropdown_value != current_selected_index:
                print_step("Callbacks", f"Route {dropdown_value} selected via dropdown.")
                result = dropdown_value
        
        if triggered_id == 'map-graph' and map_click_data:
            point = map_click_data['points'][0]
            # Base routes on the map have an integer index as customdata.
            if isinstance(point.get('customdata'), int):
                clicked_route_index = point['customdata']
                # Only update if clicking a different route
                if clicked_route_index != current_selected_index:
                    print_step("Callbacks", f"Route {clicked_route_index} selected via map click.")
                    result = clicked_route_index
        
        if result != no_update:
            route_info_ui, updated_map_figure = process_route_and_update_ui(result, current_map_figure)
            return result, route_info_ui, updated_map_figure
        
        return result, no_update, no_update

    #Not a callback

    def process_route_and_update_ui(selected_route_index, current_map_figure):
        print_step("Callbacks", f"Processing selected route index: {selected_route_index}")
        selected_route = spot.routes[selected_route_index]        
        
        processed_route = spot.get_processed_route(route_processor, 
                                                selected_route, 
                                                selected_route_index)
                    
        route_info_ui = create_route_info_card(selected_route, processed_route)
        
        # Pass the current figure and let update_map_for_selected_route handle preserving the view
        updated_map_figure = update_map_for_selected_route(current_map_figure, 
                                                           spot, 
                                                           selected_route,
                                                           processed_route)
        
        add_checkpoints(updated_map_figure, processed_route)
        
        return route_info_ui, updated_map_figure    
            
    @app.callback(
        Output('checkpoint-info', 'children'),
        Output('map-graph', 'figure', allow_duplicate=True),
        Input('map-graph', 'selectedData'),
        State('selected-route-index', 'data'),
        State('map-graph', 'figure'),
        prevent_initial_call=True
    )
    def handle_checkpoint_selection(selected_data, selected_route_index, current_figure):
        """
        Handles checkpoint selection from map clicks and updates the figure.
        """
        if not selected_data or selected_route_index is None:
            return no_update, no_update
            
        points = selected_data.get('points', [])
        if not points:
            return no_update, no_update
            
        point = points[0]
        if not isinstance(point.get('customdata'), list):
            return no_update, no_update
            
        checkpoint_index = point['customdata'][0] + 1
        selected_route = spot.routes[selected_route_index]
        processed_route = route_processor.process_route(selected_route)
        
        if checkpoint_index >= len(processed_route.checkpoints):
            return no_update, no_update
            
        checkpoint = processed_route.checkpoints[checkpoint_index]
        
        # Update the figure to highlight selected checkpoint
        fig = go.Figure(current_figure)
        for trace in fig.data:
            if trace.name == checkpoints_label:
                trace.selectedpoints = [checkpoint_index]
                break
                
        return create_checkpoint_card(checkpoint), fig

    @app.callback(
        Output('map-graph', 'figure', allow_duplicate=True),
        Input('prev-checkpoint-button', 'n_clicks'),
        Input('next-checkpoint-button', 'n_clicks'),
        State('selected-route-index', 'data'),
        State('map-graph', 'figure'),
        prevent_initial_call=True
    )
    def navigate_checkpoints(prev_clicks, next_clicks, selected_route_index, current_figure):
        """
        Handles navigation between checkpoints and updates map selection.
        """
        if selected_route_index is None:
            return no_update
            
        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        selected_route = spot.routes[selected_route_index]
        processed_route = route_processor.process_route(selected_route)
        checkpoints = processed_route.checkpoints
        
        if not checkpoints:
            return no_update
            
        # Find current selection
        current_selection = None
        for trace in current_figure['data']:
            if trace.get('name') == checkpoints_label:
                current_selection = trace.get('selectedpoints', [None])[0]
                break
                
        # Calculate new index
        if triggered_id == 'prev-checkpoint-button':
            new_index = current_selection - 1 if current_selection is not None else len(checkpoints) - 1
        else:
            new_index = current_selection + 1 if current_selection is not None else 0
        new_index %= len(checkpoints)
        
        # Update figure with new selection
        fig = go.Figure(current_figure)
        for trace in fig.data:
            if trace.name == checkpoints_label:
                trace.selectedpoints = [new_index]
                break
                
        return fig

    # Add this new callback to update the checkpoint info when navigating
    @app.callback(
        Output('checkpoint-info', 'children', allow_duplicate=True),
        Input('map-graph', 'figure'),
        State('selected-route-index', 'data'),
        prevent_initial_call=True
    )
    def update_checkpoint_info_from_figure(figure, selected_route_index):
        if selected_route_index is None:
            return no_update
            
        # Find selected checkpoint from figure
        selected_index = None
        for trace in figure['data']:
            if trace.get('name') == checkpoints_label:
                selected_index = trace.get('selectedpoints', [None])[0]
                break
                
        if selected_index is None:
            return no_update
            
        selected_route = spot.routes[selected_route_index]
        processed_route = route_processor.process_route(selected_route)
        
        if selected_index >= len(processed_route.checkpoints):
            return no_update
            
        checkpoint = processed_route.checkpoints[selected_index]
        return create_checkpoint_card(checkpoint)