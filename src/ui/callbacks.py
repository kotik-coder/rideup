# callbacks.py
from dash import ClientsideFunction, Input, Output, State, callback_context, ctx, no_update, html
from dash import MATCH
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
                                                           route_profiles['profile'])
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
                if hasattr(trace, 'name') and trace['name'] == checkpoints_label:
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

    # callbacks.py (updated section)
    @app.callback(
        Output('profile-graph', 'figure'),
        Output('bottom-panel-container', 'style', allow_duplicate=True),
        Input('selected-route-index', 'data'),
        Input('selected-checkpoint-index', 'data'),
        Input('graph-selector', 'value'),
        State('bottom-panel-container', 'style'),
        prevent_initial_call=True
    )
    def update_profile_graph(route_index, cp_index, graph_type, current_style):
        """
        Updates both the graph content and maintains its style
        """
        if route_index is None:
            return go.Figure(), {'height': '100%', 'width': '100%'}
            
        selected_route = spot.routes[route_index]
        processed_route = spot.get_processed_route(route_processor, selected_route, route_index)
        
        profiles = generate_route_profiles(
            spot,
            processed_route, 
            [t for t in spot.tracks if t.route == selected_route]
        )
        
        highlight_distance = None
        if (cp_index is not None 
            and processed_route.checkpoints 
            and cp_index < len(processed_route.checkpoints)):
            highlight_distance = processed_route.checkpoints[cp_index].distance_from_origin
        
        if graph_type == 'elevation':
            fig = create_elevation_profile_figure(profiles['profile'], highlight_distance)
        else:
            # Handle the new velocity profile format
            velocity_data = profiles['dynamic']
            
            # Check if we have both actual and theoretical data
            actual_points = velocity_data.get('actual', [])
            theoretical_points = None
            
            # For velocity graph, we can show either comparison or individual theoretical profile
            if 'theoretical' in velocity_data:
                theoretical_points = velocity_data['theoretical']['analysis_points']
            
            fig = create_velocity_profile_figure(
                actual_profile_points=actual_points,
                theoretical_profile_points=theoretical_points,
                highlight_distance=highlight_distance
            )
            
        current_style.update({
            'height': '25vh',  # Fixed height when visible
            'display': 'flex',  # Ensure flex layout is maintained
            'flex-direction': 'column',
        })
        
        return fig, current_style

    @app.callback(
        Output('right-panel', 'style'),
        Output('right-panel-toggle', 'children'),
        Output('spot-info-card', 'style'),
        Output('spot-info-toggle', 'children'),
        Output('route-info-card', 'style'),
        Output('route-info-toggle', 'children'),
        Output('bottom-panel-container', 'style'),
        Output('bottom-panel-toggle', 'children'),
        Output('right-panel-state', 'data'),
        Output('spot-info-state', 'data'),
        Output('route-info-state', 'data'),
        Output('bottom-panel-state', 'data'),
        Input('right-panel-toggle', 'n_clicks'),
        Input('spot-info-toggle', 'n_clicks'),
        Input('route-info-toggle', 'n_clicks'),
        Input('bottom-panel-toggle', 'n_clicks'),
        State('right-panel-state', 'data'),
        State('spot-info-state', 'data'),
        State('route-info-state', 'data'),
        State('bottom-panel-state', 'data'),
        State('bottom-panel-container', 'style'),
        prevent_initial_call=True
    )
    def handle_all_panel_toggles(
        right_clicks, spot_clicks, route_clicks, bottom_clicks,
        right_state, spot_state, route_state, bottom_state,
        bottom_panel_style
    ):
        # Constants for header heights (right panels only)
        HEADER_HEIGHT = '50px'
        HEADER_HEIGHT_PX = 50
        
        # Get current visibility states
        right_visible = right_state['visible']
        spot_visible = spot_state['visible']
        route_visible = route_state['visible']
        bottom_visible = bottom_state['visible']

        # Determine which button was clicked
        triggered_id = ctx.triggered_id if ctx.triggered else None
        
        # Update visibility states
        if triggered_id == 'right-panel-toggle':
            right_visible = not right_visible
            if right_visible and not (spot_visible or route_visible):
                spot_visible = True
                route_visible = True
        elif triggered_id == 'spot-info-toggle':
            spot_visible = not spot_visible
        elif triggered_id == 'route-info-toggle':
            route_visible = not route_visible
        elif triggered_id == 'bottom-panel-toggle':
            bottom_visible = not bottom_visible

        # Calculate right panel heights (with persistent headers)
        if spot_visible and route_visible:
            spot_height = f'calc(50% - {HEADER_HEIGHT_PX/2}px)'
            route_height = f'calc(50% - {HEADER_HEIGHT_PX/2}px)'
        elif spot_visible:
            spot_height = f'calc(100% - {HEADER_HEIGHT_PX}px)'
            route_height = HEADER_HEIGHT
        elif route_visible:
            spot_height = HEADER_HEIGHT
            route_height = f'calc(100% - {HEADER_HEIGHT_PX}px)'
        else:
            spot_height = HEADER_HEIGHT
            route_height = HEADER_HEIGHT

        # Right panel style - stretches fully to bottom
        right_style = {
            'position': 'fixed',
            'top': '20px',
            'right': '20px' if right_visible else '-35%',
            'width': '35%',
            'max-width': '600px',
            'z-index': '1',
            'bottom': '20px',  # Changed from 30vh to fixed 20px
            'transition': 'all 0.3s ease'
        }
        right_icon = html.I(className="fas fa-chevron-left" if right_visible else "fas fa-chevron-right")

        # Right panel cards with persistent headers
        spot_style = {
            'background-color': 'rgba(255, 255, 255, 0.85)',
            'margin-bottom': '10px',
            'height': spot_height,
            'overflow-y': 'auto' if spot_visible else 'hidden',
            'transition': 'all 0.3s ease',
            'border-top': '2px solid #ddd',
            'position': 'relative'
        }
        spot_icon = html.I(className="fas fa-minus" if spot_visible else "fas fa-plus")

        route_style = {
            'background-color': 'rgba(255, 255, 255, 0.85)',
            'height': route_height,
            'overflow-y': 'auto' if route_visible else 'hidden',
            'transition': 'all 0.3s ease',
            'border-top': '2px solid #ddd',
            'position': 'relative'
        }
        route_icon = html.I(className="fas fa-minus" if route_visible else "fas fa-plus")

        # Bottom panel - completely hides when minimized
        bottom_style = {
            'position': 'fixed',
            'bottom': '20px',
            'left': '20px',
            'right': f'calc(35% + 40px)' if right_visible else '20px',
            'height': '25vh' if bottom_visible else '0',
            'background-color': 'rgba(255, 255, 255, 0.85)',
            'border-radius': '8px',
            'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
            'z-index': '1',
            'overflow': 'hidden',
            'transition': 'all 0.3s ease',
            'display': 'flex' if bottom_visible else 'none',  # Completely hide when minimized
            'flex-direction': 'column'
        }
        bottom_icon = html.I(className="fas fa-chevron-down" if bottom_visible else "fas fa-chevron-up")

        return (
            right_style, right_icon,
            spot_style, spot_icon,
            route_style, route_icon,
            bottom_style, bottom_icon,
            {'visible': right_visible},
            {'visible': spot_visible},
            {'visible': route_visible},
            {'visible': bottom_visible}
        )
                
    @app.callback(
        Output({"type": "checkpoint-photo-modal", "index": MATCH}, "is_open"),
        Input({"type": "checkpoint-photo-thumbnail", "index": MATCH}, "n_clicks"),
        State({"type": "checkpoint-photo-modal", "index": MATCH}, "is_open"),
        prevent_initial_call=True
    )
    def toggle_photo_modal(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open

    @app.callback(
        Output('checkpoint-info', 'children', allow_duplicate=True),
        Output('selected-checkpoint-index', 'data', allow_duplicate=True),
        Output('map-graph', 'figure', allow_duplicate=True),
        Input('checkpoint-selector', 'value'),
        State('selected-route-index', 'data'),
        State('map-graph', 'figure'),
        prevent_initial_call=True
    )
    def handle_checkpoint_dropdown(selected_index, route_index, current_figure):
        if selected_index is None or route_index is None:
            return no_update, no_update, no_update
            
        selected_route = spot.routes[route_index]
        processed_route = spot.get_processed_route(route_processor, selected_route, route_index)
        
        if selected_index >= len(processed_route.checkpoints):
            return no_update, no_update, no_update
        
        checkpoint = processed_route.checkpoints[selected_index]
        
        # Update the map figure
        fig = go.Figure(current_figure)
        select_checkpoint(fig, selected_index)
        
        return create_checkpoint_card(checkpoint, processed_route), selected_index, fig