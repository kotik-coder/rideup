# callbacks.py
from dash import ClientsideFunction, Input, Output, State, callback_context, no_update, html
import dash

from src.routes import spot
from src.ui.map_helpers import print_step
from src.ui.map_visualization import *
from src.routes.route_processor import RouteProcessor
from src.routes.spot import Spot

from src.ui.ui_components import create_checkpoint_card, create_route_info_card, create_spot_info_card
from src.ui.graph_generation import create_elevation_profile_figure, create_velocity_profile_figure
from src.routes.statistics_collector import generate_route_profiles

def setup_callbacks(app, spot: Spot, route_processor: RouteProcessor):
    
    app.clientside_callback(
        ClientsideFunction(
            namespace='clientside',
            function_name='getDimensions'
        ),
        Output('map-dimensions-store', 'data'),
        Input('map-graph', 'id') # Trigger once on page load
    )
    
    """
    Registers all Dash callbacks with the provided Dash app instance.
    Handles route selection from both the dropdown and map clicks.
    """

    @app.callback(
        Output('route-selector', 'options'),
        Input('initial-load-trigger', 'children')
    )
    def initialize_dropdown(initial_load):
        """
        Populates the dropdown options on initial load only.
        """
        options = [{'label': route.name, 'value': i} for i, route in enumerate(spot.routes)]
        return options if spot.routes else []

    @app.callback(
        Output('spot-general-info', 'children'),
        Input('initial-load-trigger', 'children')
    )
    def initialize_spot_info(initial_load):
        return create_spot_info_card(spot)

    # Update the route selection callback to use the simplified route info card
    @app.callback(
        Output('selected-route-index', 'data'),
        Output('route-general-info', 'children', allow_duplicate=True),
        Output('checkpoint-info', 'children', allow_duplicate=True),
        Output('map-graph', 'figure', allow_duplicate=True),
        Input('route-selector', 'value'),
        Input('map-graph', 'clickData'),
        State('map-graph', 'figure'),
        [State('map-dimensions-store', 'data')],
        prevent_initial_call=True
    )
    def handle_route_selection(dropdown_value, map_click_data, current_map_figure, map_dims):
        """
        Handles route selection from both dropdown and map clicks.
        Map clicks update the dropdown value through a separate callback.
        """
        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

        if triggered_id == 'route-selector' and dropdown_value is not None:
            print_step("Callbacks", f"Route {dropdown_value} selected via dropdown.")
            return process_route_selection(dropdown_value, current_map_figure, map_dims)
        
        if triggered_id == 'map-graph' and map_click_data:
            point = map_click_data['points'][0]
            if isinstance(point.get('customdata'), int):
                clicked_route_index = point['customdata']
                print_step("Callbacks", f"Route {clicked_route_index} selected via map click.")
                # Return the selection but don't update dropdown here
                return process_route_selection(clicked_route_index, current_map_figure, map_dims)

        return no_update, no_update, no_update, no_update

    @app.callback(
        Output('route-selector', 'value'),
        Input('map-graph', 'clickData'),
        State('map-graph', 'figure'),
        prevent_initial_call=True
    )
    def update_dropdown_from_map(map_click_data, current_figure):

        if not map_click_data:
            return no_update

        point        = map_click_data['points'][0]
        
        '''each point is assigned the same identifying number as the route
        see map_visualization.py'''            
            
        if 'customdata' in point:
            data = point['customdata']
            if type(data) is int:
                return data
            
        return no_update
    
    def process_route_selection(selected_route_index, current_map_figure, map_dims):
        """
        Helper function to process route selection and generate UI components.
        """
        print_step("Callbacks", f"Processing selected route index: {selected_route_index}")
        selected_route = spot.routes[selected_route_index]
        processed_route = spot.get_processed_route(route_processor, selected_route, selected_route_index)

        route_profiles = generate_route_profiles(
            spot,
            processed_route, 
            [t for t in spot.tracks if t.route == selected_route]
        )
        route_info_ui = create_route_info_card(selected_route, processed_route, route_profiles)
        updated_map_figure = update_map_for_selected_route(current_map_figure, 
                                                           spot, 
                                                           selected_route, 
                                                           processed_route, 
                                                           map_dims, 
                                                           route_profiles)
        add_checkpoints(updated_map_figure, processed_route)

        checkpoint_info_ui = html.Div()
        if processed_route.checkpoints:
            first_checkpoint = processed_route.checkpoints[0]
            checkpoint_info_ui = create_checkpoint_card(first_checkpoint, processed_route)
            select_checkpoint(updated_map_figure, 0)

        return selected_route_index, route_info_ui, checkpoint_info_ui, updated_map_figure
    
    def select_checkpoint(map_figure, cp_index : int):
        if 'data' in map_figure:
            for trace in map_figure['data']:
                if hasattr(trace, 'name') and trace.name == checkpoints_label:
                    trace.selectedpoints = [cp_index]
                    break
    
    @app.callback( 
        Output('checkpoint-info', 'children', allow_duplicate=True),
        Output('selected-checkpoint-index', 'data'), 
        Input('map-graph', 'selectedData'),
        State('selected-route-index', 'data'),
        prevent_initial_call=True
    )
    def handle_checkpoint_clicks(selected_data, selected_route_index):
        """
        Handles checkpoint selection from map clicks only.
        """
        if not selected_data or selected_route_index is None:
            return no_update, no_update
         
        points = selected_data.get('points', [])
        if points:            

            point = points[0]

            if isinstance(point.get('customdata'), list):
               
                checkpoint_index = point['customdata'][0]
                selected_route  = spot.routes[selected_route_index]
                processed_route = spot.get_processed_route(route_processor, selected_route, selected_route_index)
                
                checkpoint = processed_route.checkpoints[checkpoint_index]
                return create_checkpoint_card(checkpoint, processed_route), checkpoint_index
            
        return no_update, no_update

    @app.callback(
        Output('checkpoint-info', 'children', allow_duplicate=True),
        Output('selected-checkpoint-index', 'data', allow_duplicate=True), # Output to update the store
        Output('map-graph', 'figure', allow_duplicate=True),
        Input('prev-checkpoint-button', 'n_clicks'),
        Input('next-checkpoint-button', 'n_clicks'),
        State('selected-route-index', 'data'),
        State('selected-checkpoint-index', 'data'), # Read current checkpoint index from store
        State('map-graph', 'figure'),
        prevent_initial_call=True
    )
    def handle_checkpoint_navigation(prev_clicks, next_clicks, selected_route_index, cid, mapf):
        """
        Handles navigation between checkpoints using 'Previous' and 'Next' buttons.
        """

        if selected_route_index is None or cid is None:
            return no_update, no_update, no_update        

        triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
                
        selected_route  = spot.routes[selected_route_index]
        processed_route = spot.get_processed_route(route_processor,
                                                selected_route,
                                                selected_route_index)        

        # Get the current selected checkpoint index from the dcc.Store
        # If the store is None (e.g., on initial load before any selection), default to 0.
        
        new_checkpoint_index = cid

        if triggered_id   == 'prev-checkpoint-button' and prev_clicks and prev_clicks > 0:
            new_checkpoint_index = cid - 1
        elif triggered_id == 'next-checkpoint-button' and next_clicks and next_clicks > 0:
            new_checkpoint_index = cid + 1

        # Only update if the index has changed
        if new_checkpoint_index != cid:            

            print_step("Callbacks", f"Navigating to checkpoint index: {new_checkpoint_index}")

            fig = go.Figure(mapf)
            select_checkpoint(fig, new_checkpoint_index)
            new_checkpoint = processed_route.checkpoints[new_checkpoint_index]
            
            # Return the updated checkpoint card, the figure, and the new checkpoint index to store
            return create_checkpoint_card(new_checkpoint, processed_route), new_checkpoint_index, fig
        
        return no_update, no_update, no_update
                
    @app.callback(
        Output('profile-graph', 'figure'),
        Input('selected-route-index', 'data'),
        Input('selected-checkpoint-index', 'data'),
        Input('graph-selector', 'value'),
        Input('initial-state-trigger', 'data'),
        State('selected-route-index', 'data'),
        prevent_initial_call=True
    )
    def update_profile_graph(route_trigger, cp_trigger, graph_type, initial_trigger, selected_route_index):
        """
        Updates the profile graph based on the selected graph type.
        """
        ctx = dash.callback_context
        
        # Initial state handling
        if not ctx.triggered or ctx.triggered[0]['prop_id'] == 'initial-state-trigger.data':
            empty_fig = go.Figure()
            empty_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=250,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            return empty_fig
            
        if selected_route_index is None:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=250,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            return empty_fig
        
        # Get route data
        selected_route = spot.routes[selected_route_index]
        processed_route = spot.get_processed_route(route_processor, selected_route, selected_route_index)
        
        # Generate profiles
        profiles = generate_route_profiles(
            spot,
            processed_route, 
            [t for t in spot.tracks if t.route == selected_route]
        )
        
        # Get highlight distance if checkpoint is selected
        highlight_distance = None
        if (cp_trigger is not None 
            and processed_route.checkpoints 
            and cp_trigger < len(processed_route.checkpoints)):
            highlight_distance = processed_route.checkpoints[cp_trigger].distance_from_origin
        
        # Create the appropriate figure based on selection
        if graph_type == 'elevation':
            return create_elevation_profile_figure(
                profiles['static'], 
                highlight_distance
            )
        else:  # velocity
            return create_velocity_profile_figure(
                profiles['dynamic'], 
                highlight_distance
            )