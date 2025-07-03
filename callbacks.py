import dash
from dash import Input, Output, State
import json
import numpy as np
import plotly.graph_objects as go
from map_helpers import print_step
from route import GeoPoint
from map_visualization import *
from graph_generation import *
from ui_components import create_checkpoint_card


def setup_callbacks(app, spot, spot_loader, route_processor):
    """
    Registers all Dash callbacks with the provided Dash app instance.
    Requires the Spot, SpotLoader, and RouteProcessor instances for data processing.
    """

    @app.callback(
        Output('route-data-store', 'data'),
        Output('route-selector', 'options'),
        Input('initial-load-trigger', 'children')
    )
    def load_initial_route_data(_):
        """Loads route data and populates route selector options on initial app load."""
        print_step("Callback", "Загружаю начальные данные маршрутов...")
        route_data = []
        options = []
        
        # Load routes and tracks through SpotLoader
        spot_loader.load_valid_routes_and_tracks()
        
        # Update RouteProcessor with loaded data
        route_processor.local_photos = spot.local_photos
        route_processor.route_to_tracks = spot._route_to_tracks
        
        if spot.routes:
            print_step("Callback", f"Найдено {len(spot.routes)} маршрутов для обработки.")
            for i, route in enumerate(spot.routes):
                try:
                    route_dict = route_processor.process_route(route)
                    if route_dict and route_dict.get('smooth_points') and route_dict.get('checkpoints'):
                        route_data.append(route_dict)
                        options.append({'label': route.name, 'value': i})
                        print_step("Callback", f"Маршрут '{route.name}' успешно обработан и добавлен.")
                    else:
                        print_step("Callback", f"Маршрут '{route.name}' обработан, но данные пусты или неполные. Пропускаю.", level="WARN")
                except Exception as e:
                    print_step("Callback", f"Ошибка при обработке маршрута '{route.name}': {e}", level="ERROR")

        print_step("Callback", f"Всего обработано и добавлено {len(route_data)} маршрутов.")
        return json.dumps(route_data), options

    @app.callback(
        Output('selected-route-index', 'data'),
        Output('selected-checkpoint-index', 'data'),
        Input('map-graph', 'clickData'),
        Input('route-selector', 'value'),
        State('route-data-store', 'data'),
        State('selected-route-index', 'data')
    )
    def update_selection(click_data, dropdown_value, route_data_json, current_route_index):
        """Handles selection of routes and checkpoints from either the map or dropdown."""
        triggered_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

        if triggered_id == 'route-selector':
            return dropdown_value, None

        if triggered_id == 'map-graph' and click_data:
            point_data = click_data['points'][0]

            if 'customdata' in point_data and point_data['customdata'] is not None:
                return current_route_index, point_data['customdata']

            clicked_lat = point_data['lat']
            clicked_lon = point_data['lon']
            if not route_data_json: return dash.no_update, dash.no_update
            route_data = json.loads(route_data_json)

            min_distance = float('inf')
            closest_route_index = None

            for i, route_dict in enumerate(route_data):
                for p in route_dict['smooth_points']:
                    dist = GeoPoint(clicked_lat, clicked_lon).distance_to(GeoPoint(p[0], p[1]))
                    if dist < min_distance:
                        min_distance = dist
                        closest_route_index = i

            if closest_route_index is not None and min_distance < 100:
                return closest_route_index, None

        return dash.no_update, dash.no_update

    @app.callback(
        Output('map-graph', 'figure'),
        Input('selected-route-index', 'data'),
        Input('selected-checkpoint-index', 'data'),
        State('route-data-store', 'data'),
        State('map-graph', 'figure')
    )
    def update_map_figure(route_index, checkpoint_index, route_data_json, current_figure):
        """Draws all routes and correctly zooms based on selection changes."""
        fig = go.Figure(current_figure)

        if not route_data_json:
            print_step("Callback", "update_map_figure: route_data_json is empty, returning current figure.")
            return fig

        route_data = json.loads(route_data_json)
        print_step("Callback", f"update_map_figure: Processing with {len(route_data)} routes.")

        triggered_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0] if dash.callback_context.triggered else None

        if not hasattr(fig.layout, 'meta') or not isinstance(fig.layout.meta, dict):
            fig.layout.meta = {}

        prev_centered_route_index = fig.layout.meta.get('last_centered_route_index', None)
        should_update_center_and_zoom = False
        current_centered_route_index = None

        if triggered_id is None:
            should_update_center_and_zoom = True
            current_centered_route_index = route_index
            print_step("Callback", "Initial load, updating map center and zoom.")
        elif triggered_id == 'selected-route-index':
            if route_index != prev_centered_route_index:
                should_update_center_and_zoom = True
                current_centered_route_index = route_index
                print_step("Callback", "New route selected (or route deselected), updating map center and zoom.")
            else:
                print_step("Callback", "Same route re-selected, maintaining current map center and zoom.")
        elif triggered_id == 'selected-checkpoint-index':
            should_update_center_and_zoom = False
            print_step("Callback", "Checkpoint selected, maintaining current map center and zoom.")

        fig.data = []

        if not fig.layout.map.layers:
            fig.update_layout(map_style="open-street-map")

        if should_update_center_and_zoom:
            if current_centered_route_index is not None and current_centered_route_index < len(route_data):
                selected_route = route_data[current_centered_route_index]
                if selected_route['smooth_points']:
                    lats = [p[0] for p in selected_route['smooth_points']]
                    lons = [p[1] for p in selected_route['smooth_points']]

                    center_lat = (min(lats) + max(lats)) / 2
                    center_lon = (min(lons) + max(lons)) / 2
                    zoom = calculate_zoom(lats, lons) # Use imported calculate_zoom

                    fig.update_layout(
                        map_center={'lat': center_lat, 'lon': center_lon},
                        map_zoom=zoom
                    )
                    print_step("Callback", f"Zooming to selected route: {selected_route['name']}")
            else:                
                min_lon_val, min_lat_val, max_lon_val, max_lat_val = spot.bounds # Access route_manager bounds
                center_lat = (min_lat_val + max_lat_val) / 2
                center_lon = (min_lon_val + max_lon_val) / 2
                zoom = calculate_zoom([min_lat_val, max_lat_val], [min_lon_val, max_lon_val])
                fig.update_layout(
                    map_center={'lat': center_lat, 'lon': center_lon},
                    map_zoom=zoom
                )
                add_forest_boundary_and_name_to_figure(fig, spot.bounds) # Use imported function
                print_step("Callback", "Zooming to forest bounds (no route selected).")

            fig.layout.meta['last_centered_route_index'] = current_centered_route_index

        for i, route_dict in enumerate(route_data):
            if i != route_index:
                add_route_to_figure(fig, route_dict, is_selected=False) # Use imported function

        if route_index is not None and route_index < len(route_data):
            selected_route_dict = route_data[route_index]
            add_route_to_figure(
                fig,
                selected_route_dict,
                is_selected=True,
                highlight_checkpoint=checkpoint_index
            )

        return fig

    @app.callback(
        Output('route-selector', 'value'),
        Input('selected-route-index', 'data'),
        prevent_initial_call=True
    )
    def sync_route_selector(route_index):
        return route_index

    @app.callback(
        Output('route-general-info', 'children'),
        Output('elevation-profile', 'figure'),
        Output('velocity-profile', 'figure'),
        Input('selected-route-index', 'data'),
        Input('selected-checkpoint-index', 'data'),
        State('route-data-store', 'data')
    )
    def update_all_info(route_index, checkpoint_index, route_data_json):
        common_graph_layout = {
            'margin': dict(l=20, r=20, t=40, b=20),
            'height': 250,
            'showlegend': False,
            'hovermode': 'x unified'
        }

        if route_index is None or not route_data_json:
            empty_elevation_fig = go.Figure().update_layout(
                xaxis_title="Расстояние (м)", 
                yaxis_title="Высота (м)", 
                title='Профиль высот',
                **common_graph_layout
            )
            empty_velocity_fig = go.Figure().update_layout(
                xaxis_title="Расстояние (м)", 
                yaxis_title="Скорость (км/ч)", 
                title='Профиль скорости',
                **common_graph_layout
            )
            return "Выберите маршрут для просмотра информации.", empty_elevation_fig, empty_velocity_fig

        route_data = json.loads(route_data_json)
        route = route_data[route_index]

        # Route General Info
        general_info = dash.html.Div([
            dash.html.H5(route['name'], className="mb-1 fs-6"),
            dash.html.P(f"Длина маршрута: {route['elevation_profile'][-1]['distance']:.2f} м", className="mb-1", style={'fontSize': '0.9em'}),
            dash.html.P(f"Средняя высота: {np.mean([p['elevation'] for p in route['elevation_profile']]):.1f} м", className="mb-1", style={'fontSize': '0.9em'}),
            dash.html.P(f"Набор высоты: {sum(s['elevation_gain'] for s in route['segments']):.1f} м", className="mb-1", style={'fontSize': '0.9em'}),
            dash.html.P(f"Потеря высоты: {sum(s['elevation_loss'] for s in route['segments']):.1f} м", className="mb-0", style={'fontSize': '0.9em'}),
        ])

        # Elevation Profile
        distances = [p['distance'] for p in route['elevation_profile']]
        elevations = [p['elevation'] for p in route['elevation_profile']]

        elevation_fig = go.Figure()
        elevation_fig.add_trace(go.Scatter(
            x=distances,
            y=elevations,
            mode='lines',
            line_color='blue',
            hoverinfo='text',
            hovertext=[f"Расстояние: {d:.1f} м<br>Высота: {e:.1f} м" for d, e in zip(distances, elevations)],
            showlegend=False
        ))

        if elevations:
            median_elevation = np.median(elevations)
            polygons_above_median, polygons_below_median = get_fill_polygons(distances, elevations, median_elevation)
            for poly_coords in polygons_above_median:
                if poly_coords:
                    x_coords = [p[0] for p in poly_coords]
                    y_coords = [p[1] for p in poly_coords]
                    elevation_fig.add_trace(go.Scatter(
                        x=x_coords, y=y_coords,
                        fill='toself', fillcolor='rgba(255, 0, 0, 0.3)',
                        mode='none', showlegend=False
                    ))
            for poly_coords in polygons_below_median:
                if poly_coords:
                    x_coords = [p[0] for p in poly_coords]
                    y_coords = [p[1] for p in poly_coords]
                    elevation_fig.add_trace(go.Scatter(
                        x=x_coords, y=y_coords,
                        fill='toself', fillcolor='rgba(0, 255, 0, 0.3)',
                        mode='none', showlegend=False
                    ))

        if checkpoint_index is not None and checkpoint_index < len(route['checkpoints']):
            selected_cp = route['checkpoints'][checkpoint_index]
            elevation_fig.add_trace(go.Scatter(
                x=[selected_cp['distance_from_start']],
                y=[selected_cp['elevation']],
                mode='markers',
                marker=dict(size=12, color='gold', line=dict(width=2, color='black')),
                hoverinfo='text',
                hovertext=f"{selected_cp['name']}<br>Расстояние: {selected_cp['distance_from_start']:.1f} м<br>Высота: {selected_cp['elevation']:.1f} м",
                showlegend=False
            ))

        elevation_fig.update_layout(
            xaxis_title='Расстояние (м)',
            yaxis_title='Высота (м)',
            **common_graph_layout
        )

        # Velocity Profile (converted to km/h)
        velocity_profile_data = route.get('velocity_profile', [])
        velocity_distances = [p['distance'] for p in velocity_profile_data]
        velocities_kmh = [p['velocity'] * 3.6 for p in velocity_profile_data]

        velocity_fig = go.Figure()
        if velocity_distances and velocities_kmh:

            velocity_fig.add_trace(go.Scatter(
                x=velocity_distances,
                y=velocities_kmh,
                mode='lines',
                line_color='green',
                hoverinfo='text',
                hovertext=[f"Расстояние: {d:.1f} м<br>Скорость: {v:.1f} км/h" for d, v in zip(velocity_distances, velocities_kmh)],
                showlegend=False
            ))

            if len(velocities_kmh) > 1:
                q1_vel_kmh, median_vel_kmh, q3_vel_kmh = calculate_velocity_quartiles(velocities_kmh)

                high_vel_polygons_above_q3, _ = get_fill_polygons(velocity_distances, velocities_kmh, q3_vel_kmh)
                for poly_coords in high_vel_polygons_above_q3:
                    if poly_coords:
                        x_coords = [p[0] for p in poly_coords]
                        y_coords = [p[1] for p in poly_coords]
                        velocity_fig.add_trace(go.Scatter(
                            x=x_coords, y=y_coords,
                            fill='toself', fillcolor='rgba(255, 0, 0, 0.3)',
                            mode='none', showlegend=False
                        ))

                _, low_vel_polygons_below_q1 = get_fill_polygons(velocity_distances, velocities_kmh, q1_vel_kmh)
                for poly_coords in low_vel_polygons_below_q1:
                    if poly_coords:
                        x_coords = [p[0] for p in poly_coords]
                        y_coords = [p[1] for p in poly_coords]
                        velocity_fig.add_trace(go.Scatter(
                            x=x_coords, y=y_coords,
                            fill='toself', fillcolor='rgba(0, 0, 255, 0.3)',
                            mode='none', showlegend=False
                        ))

        if velocity_distances:
            velocity_fig.add_shape(
                type='line',
                x0=min(velocity_distances),
                y0=6,
                x1=max(velocity_distances),
                y1=6,
                line=dict(color='gray', width=1, dash='dot'),
                name='Скорость пешехода',
                layer="below"
            )
            velocity_fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                name='Скорость пешехода',
                hoverinfo='name',
                showlegend=False
            ))


        if checkpoint_index is not None and checkpoint_index < len(route['checkpoints']):
            selected_cp = route['checkpoints'][checkpoint_index]
            if velocity_profile_data:
                closest_velocity_point = min(velocity_profile_data,
                                            key=lambda p: abs(p['distance'] - selected_cp['distance_from_start']))
                velocity_kmh = closest_velocity_point['velocity'] * 3.6
                velocity_fig.add_trace(go.Scatter(
                    x=[closest_velocity_point['distance']],
                    y=[velocity_kmh],
                    mode='markers',
                    marker=dict(size=12, color='gold', line=dict(width=2, color='black')),
                    hoverinfo='text',
                    hovertext=f"{selected_cp['name']}<br>Расстояние: {closest_velocity_point['distance']:.1f} м<br>Скорость: {velocity_kmh:.1f} км/ч",
                    showlegend=False
                ))

        velocity_fig.update_layout(
            xaxis_title='Расстояние (м)',
            yaxis_title='Скорость (км/ч)',
            **common_graph_layout
        )

        velocity_fig.add_annotation(
            text="&#x3F;",
            xref="paper", yref="paper",
            x=0.95, y=0.95,
            showarrow=False,
            font=dict(size=14, color="white"),
            bgcolor="gray",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            hovertext="<i>Расчет скорости: центральная разность + медианный фильтр + адаптивный фильтр Савицкого-Голея.</i>",
            xanchor='right',
            yanchor='top'
        )

        return general_info, elevation_fig, velocity_fig

    @app.callback(
        Output('checkpoint-info', 'children'),
        Input('selected-route-index', 'data'),
        Input('selected-checkpoint-index', 'data'),
        State('route-data-store', 'data'),
        prevent_initial_call=True
    )
    def update_checkpoint_info(route_index, checkpoint_index, route_data_json):
        if route_index is None or checkpoint_index is None or not route_data_json:
            return "Выберите чекпоинт на карте, чтобы увидеть информацию."

        route_data = json.loads(route_data_json)
        if route_index >= len(route_data):
            return "Неверный индекс маршрута."

        route = route_data[route_index]
        if checkpoint_index >= len(route['checkpoints']):
            return "Неверный индекс чекпоинта."

        return create_checkpoint_card(route['checkpoints'][checkpoint_index]) # Use imported function

    @app.callback(
        Output('selected-checkpoint-index', 'data', allow_duplicate=True),
        Input('prev-checkpoint-button', 'n_clicks'),
        Input('next-checkpoint-button', 'n_clicks'),
        State('selected-route-index', 'data'),
        State('selected-checkpoint-index', 'data'),
        State('route-data-store', 'data'),
        prevent_initial_call=True
    )
    def navigate_checkpoints(prev_n_clicks, next_n_clicks, route_index, current_checkpoint_index, route_data_json):
        triggered_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

        if route_index is None or route_data_json is None:
            return dash.no_update

        route_data = json.loads(route_data_json)
        if route_index >= len(route_data):
            return dash.no_update

        current_route = route_data[route_index]
        checkpoints = current_route.get('checkpoints', [])
        total_checkpoints = len(checkpoints)

        if total_checkpoints == 0:
            return dash.no_update

        new_checkpoint_index = current_checkpoint_index if current_checkpoint_index is not None else 0

        if triggered_id == 'prev-checkpoint-button':
            new_checkpoint_index = max(0, new_checkpoint_index - 1)
        elif triggered_id == 'next-checkpoint-button':
            new_checkpoint_index = min(total_checkpoints - 1, new_checkpoint_index + 1)

        if new_checkpoint_index != current_checkpoint_index:
            return new_checkpoint_index
        return dash.no_update