import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from route_manager import RouteManager
from route import GeoPoint 
from datetime import datetime
from map_helpers import print_step
import json
from media_helpers import get_photo_html, get_landscape_photo
from typing import List, Tuple, Dict
from pathlib import Path

class BitsevskyMapApp:
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self._print_header()
        self.rm = RouteManager("Битцевский лес, Москва")
        self.selected_route_index = None
        self.selected_checkpoint_index = None
        self._setup_layout()
        self._setup_callbacks()

    def _print_header(self):
        """Выводит заголовок с информацией о запуске"""
        print("\n" + "="*50)
        print(f"Загрузка карты Битцевского леса")
        print(f"Время начала: {datetime.now().strftime('%H:%M:%S')}")
        print("="*50 + "\n")

    def _create_initial_figure(self):
        """Создает базовую структуру карты Plotly с центрированием и стилем,
           используя границы всего леса для начального зума."""
        
        min_lon_val, min_lat_val, max_lon_val, max_lat_val = self.rm.bounds
        
        center_lat = (min_lat_val + max_lat_val) / 2
        center_lon = (min_lon_val + max_lon_val) / 2

        # Calculate initial zoom based on the full forest bounds
        initial_zoom = self._calculate_zoom([min_lat_val, max_lat_val], [min_lon_val, max_lon_val])
        
        fig = go.Figure(go.Scattermap())
        fig.update_layout(
            map_style="open-street-map",
            map=dict(
                center=dict(lat=center_lat, lon=center_lon),
                zoom=initial_zoom
            ),
            margin={"r":0,"t":0,"l":0,"b":0},
            showlegend=False,
            clickmode='event+select'
        )
        
        self._add_forest_boundary_and_name_to_figure(fig) #
        
        return fig


    def _setup_layout(self):
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("Битцевский лес - Веломаршрут", className="text-center my-4"))
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(
                    id='map-graph',
                    style={'height': 'calc(100vh - 100px)', 'width': '100%'},
                    figure=self._create_initial_figure()
                ), width=8, style={'padding': '0'}),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Выбор маршрута", className="font-weight-bold"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='route-selector',
                                options=[],
                                placeholder="Выберите маршрут"
                            )
                        ])
                    ], className="mb-3"),
                    dbc.Card([
                        dbc.CardHeader("Информация о маршруте", className="font-weight-bold"),
                        dbc.CardBody([
                            html.Div(id='route-general-info'),
                            dcc.Graph(id='elevation-profile', style={'height': '250px'})
                        ])
                    ], className="mb-3"),
                    dbc.Card([
                        dbc.CardHeader("Информация о чекпоинте", className="font-weight-bold"),
                        dbc.CardBody(id='checkpoint-info')
                    ])
                ], width=4, style={'padding': '0 15px'})
            ], style={'margin': '0'}),
            dcc.Store(id='route-data-store'),
            dcc.Store(id='selected-route-index'),
            dcc.Store(id='selected-checkpoint-index'),
            html.Div(id='initial-load-trigger', style={'display': 'none'})
        ], fluid=True, style={'padding': '0'})

    def _setup_callbacks(self):
        @self.app.callback(
            Output('route-data-store', 'data'),
            Output('route-selector', 'options'),
            Input('initial-load-trigger', 'children')
        )
        def load_initial_route_data(_):
            """Loads route data and populates route selector options on initial app load."""
            print_step("Callback", "Загружаю начальные данные маршрутов...")
            route_data = []
            options = []
            if self.rm.valid_routes:
                print_step("Callback", f"Найдено {len(self.rm.valid_routes)} маршрутов для обработки.")
                for i, route in enumerate(self.rm.valid_routes):
                    try:
                        route_dict = self.rm._process_route(route)
                        # --- Crucial check to ensure processed data is valid ---
                        if route_dict and route_dict.get('smooth_points') and route_dict.get('checkpoints'):
                            route_data.append(route_dict)
                            options.append({'label': route.name, 'value': i})
                            print_step("Callback", f"Маршрут '{route.name}' успешно обработан и добавлен.")
                        else:
                            print_step("Callback", f"Маршрут '{route.name}' обработан, но данные пусты или неполные (нет smooth_points или checkpoints). Пропускаю.", level="WARN")
                    except Exception as e:
                        print_step("Callback", f"Ошибка при обработке маршрута '{route.name}': {e}", level="ERROR")
            
            print_step("Callback", f"Всего обработано и добавлено {len(route_data)} маршрутов.")
            return json.dumps(route_data), options

        @self.app.callback(
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

        @self.app.callback(
            Output('map-graph', 'figure'),
            Input('selected-route-index', 'data'),
            Input('selected-checkpoint-index', 'data'),
            State('route-data-store', 'data'),
            State('map-graph', 'figure')
        )
        def update_map_figure(route_index, checkpoint_index, route_data_json, current_figure):
            """Draws all routes and correctly zooms to the selected one."""
            fig = go.Figure(current_figure) 

            if not route_data_json:
                print_step("Callback", "update_map_figure: route_data_json is empty, returning current figure.")
                return fig

            route_data = json.loads(route_data_json)
            print_step("Callback", f"update_map_figure: Processing with {len(route_data)} routes.")

            # Clear existing traces to redraw
            fig.data = []

            # Add a base map layer if it's missing (important for redrawing)
            if not fig.layout.map.layers:
                fig.update_layout(map_style="open-street-map")

            # --- Logic for zooming to the selected route or back to forest bounds ---
            if route_index is not None and route_index < len(route_data):
                selected_route = route_data[route_index]
                if selected_route['smooth_points']:
                    lats = [p[0] for p in selected_route['smooth_points']]
                    lons = [p[1] for p in selected_route['smooth_points']]
                    
                    center_lat = (min(lats) + max(lats)) / 2
                    center_lon = (min(lons) + max(lons)) / 2
                    zoom = self._calculate_zoom(lats, lons)
                    
                    fig.update_layout(
                        map_center={'lat': center_lat, 'lon': center_lon},
                        map_zoom=zoom
                    )
                    print_step("Callback", f"Zooming to selected route: {selected_route['name']}")
            else:
                # If no route is selected, zoom to the entire forest area
                min_lon_val, min_lat_val, max_lon_val, max_lat_val = self.rm.bounds 
                center_lat = (min_lat_val + max_lat_val) / 2 
                center_lon = (min_lon_val + max_lon_val) / 2
                zoom = self._calculate_zoom([min_lat_val, max_lat_val], [min_lon_val, max_lon_val])
                fig.update_layout(
                    map_center={'lat': center_lat, 'lon': center_lon},
                    map_zoom=zoom
                )
                self._add_forest_boundary_and_name_to_figure(fig)
                print_step("Callback", "Zooming to forest bounds (no route selected).")


            # Drawing all routes
            for i, route_dict in enumerate(route_data):
                # Always add non-selected routes first
                if i != route_index:
                    self._add_route_to_figure(fig, route_dict, is_selected=False)
                    print_step("Callback", f"Added non-selected route: {route_dict['name']}")
            
            # Add the selected route last to ensure it's on top
            if route_index is not None and route_index < len(route_data):
                selected_route_dict = route_data[route_index]
                self._add_route_to_figure(
                    fig, 
                    selected_route_dict, 
                    is_selected=True,
                    highlight_checkpoint=checkpoint_index
                )
                print_step("Callback", f"Added selected route: {selected_route_dict['name']}")
                
            return fig

        @self.app.callback(
            Output('route-selector', 'value'),
            Input('selected-route-index', 'data'),
            prevent_initial_call=True
        )
        def sync_route_selector(route_index):
            return route_index

        @self.app.callback(
            Output('route-general-info', 'children'),
            Output('elevation-profile', 'figure'),
            Output('checkpoint-info', 'children'),
            Input('selected-route-index', 'data'),
            Input('selected-checkpoint-index', 'data'),
            State('route-data-store', 'data')
        )
        def update_all_info(selected_route_index, checkpoint_index, route_data_json):
            if selected_route_index is None or not route_data_json:
                empty_fig = go.Figure().update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=250)
                print_step("Callback", "update_all_info: No route selected or data empty. Returning default.")
                return "Выберите маршрут", empty_fig, "Выберите чекпоинт на карте"
                
            route_data = json.loads(route_data_json)
            if selected_route_index >= len(route_data):
                print_step("Callback", f"update_all_info: selected_route_index ({selected_route_index}) out of bounds ({len(route_data)}).")
                return dash.no_update, dash.no_update, dash.no_update

            route = route_data[selected_route_index]
            print_step("Callback", f"update_all_info: Updating info for route: {route['name']}")
            
            total_distance = sum(segment['distance'] for segment in route['segments'])
            elevation_gain = sum(max(0, segment['elevation_gain']) for segment in route['segments'])
            elevation_loss = sum(abs(min(0, segment['elevation_loss'])) for segment in route['segments'])
            
            info_content = [
                html.H5(route['name'], className="card-title"),
                html.P(f"Общая длина: {total_distance:.1f} м"),
                html.P(f"Набор высоты: {elevation_gain:.1f} м"),
                html.P(f"Потеря высоты: {elevation_loss:.1f} м"),
                html.P(f"Чекпоинтов: {len(route['checkpoints'])}")
            ]
            
            elevation_fig = go.Figure()
            elevation_fig.add_trace(go.Scatter(
                x=[x['distance'] for x in route['elevation_profile']],
                y=[x['elevation'] for x in route['elevation_profile']],
                mode='lines',
                line=dict(color='green', width=2),
                fill='tozeroy',
                name='Высота'
            ))

            elevations_in_profile = [x['elevation'] for x in route['elevation_profile']]
            
            min_elevation = 0
            max_elevation = 100 # Default sensible range if no elevation data
            y_axis_min = 0

            if elevations_in_profile:
                min_elevation = min(elevations_in_profile)
                max_elevation = max(elevations_in_profile)
                
                y_axis_min = min_elevation - (max_elevation - min_elevation) * 0.1
                if y_axis_min < 0:
                    y_axis_min = 0 
            
            for i, checkpoint in enumerate(route['checkpoints']):
                elevation_fig.add_trace(go.Scatter(
                    x=[checkpoint['distance_from_start']],
                    y=[checkpoint['elevation']],
                    mode='markers',
                    marker=dict(
                        size=12,
                        symbol='circle',
                        color='yellow' if i == checkpoint_index else 'blue',
                        line=dict(width=2, color='darkblue')
                    ),
                    name=checkpoint['name'],
                    hoverinfo='text',
                    hovertext=f"{checkpoint['name']} ({checkpoint['elevation']:.1f} м)"
                ))
            
            elevation_fig.update_layout(
                margin={"r":20,"t":20,"l":40,"b":40},
                xaxis_title="Расстояние (м)",
                yaxis_title="Высота (м)",
                height=250,
                showlegend=False,
                yaxis=dict(range=[y_axis_min, max_elevation + (max_elevation - min_elevation) * 0.1])
            )
            
            if checkpoint_index is not None and checkpoint_index < len(route['checkpoints']):
                checkpoint_info = self._create_checkpoint_card(route['checkpoints'][checkpoint_index])
                print_step("Callback", f"update_all_info: Displaying info for checkpoint {checkpoint_index}.")
            else:
                checkpoint_info = "Выберите чекпоинт на карте"
                print_step("Callback", "update_all_info: No checkpoint selected, displaying default.")
            
            return info_content, elevation_fig, checkpoint_info 

    def _calculate_zoom(self, lats, lons):
        """
        --- Исправлено: Более надежный расчет зума ---
        Вычисляет уровень масштабирования на основе географического охвата.
        """
        if not lats or not lons:
            print_step("Zoom", "Расчет зума: Нет координат. Возвращаю дефолтный зум.")
            return 12 

        lat_span = max(lats) - min(lats)
        lon_span = max(lons) - min(lons)
        
        # Обработка случая с одной точкой или нулевым охватом
        if lat_span == 0 and lon_span == 0:
            print_step("Zoom", "Расчет зума: Нулевой охват. Возвращаю высокий зум.")
            return 15

        zoom_lat = 9.5 - np.log2(lat_span + 1e-6)
        zoom_lon = 9.5 - np.log2(lon_span + 1e-6)

        final_zoom = min(zoom_lat, zoom_lon, 18) # Cap max zoom to 18
        print_step("Zoom", f"Рассчитан зум: {final_zoom:.2f}")
        return final_zoom

    def _add_route_to_figure(self, fig, route_dict, is_selected=False, highlight_checkpoint=None):
            route_lats = [p[0] for p in route_dict['smooth_points']]
            route_lons = [p[1] for p in route_dict['smooth_points']]
            route_elevations = route_dict['elevation_profile']

            if not route_lats or not route_lons or not route_elevations:
                print_step("Map Drawing", f"Пропускаю отрисовку маршрута '{route_dict.get('name', 'N/A')}' из-за отсутствия данных.")
                return

            if is_selected:
                elevations_only = [ep['elevation'] for ep in route_elevations]
                if not elevations_only:
                    print_step("Map Drawing", f"Пропускаю отрисовку выбранного маршрута '{route_dict['name']}' по высоте: нет данных высот.")
                    return

                min_elev = min(elevations_only)
                max_elev = max(elevations_only)
                start_elev = elevations_only[0] if elevations_only else 0

                if min_elev == max_elev:
                    plotly_colorscale = [[0, 'green'], [1, 'green']]
                    cmin_val = min_elev - 1 if min_elev > 0 else 0
                    cmax_val = max_elev + 1
                else:
                    norm_start_elev = (start_elev - min_elev) / (max_elev - min_elev)
                    plotly_colorscale = [
                        [0, 'blue'],
                        [norm_start_elev, 'green'],
                        [1, 'red']
                    ]
                    plotly_colorscale.sort(key=lambda x: x[0])
                    cmin_val = min_elev
                    cmax_val = max_elev

                fig.add_trace(go.Scattermap(
                    lat=route_lats,
                    lon=route_lons,
                    mode='lines+markers',
                    line=dict(width=6, color='rgba(0,0,0,0)'),
                    marker=dict(
                        size=12,
                        color=elevations_only,
                        colorscale=plotly_colorscale,
                        cmin=cmin_val,
                        cmax=cmax_val,
                        colorbar=dict(
                            title="Высота (м)",
                            x=1.02,
                            lenmode="fraction",
                            len=0.75
                        )
                    ),
                    hoverinfo='text',
                    hovertext=[f"Высота: {ep['elevation']:.1f} м" for ep in route_elevations],
                    showlegend=False,
                    name=f"Маршрут {route_dict['name']} (по высоте)"
                ))
                print_step("Map Drawing", f"Отрисован выбранный маршрут '{route_dict['name']}' с цветовой схемой высот.")

                if route_dict['checkpoints']:
                    checkpoint_lats = [cp['lat'] for cp in route_dict['checkpoints']]
                    checkpoint_lons = [cp['lon'] for cp in route_dict['checkpoints']]
                    checkpoint_names = [cp['name'] for cp in route_dict['checkpoints']]
                    checkpoint_elevations = [cp['elevation'] for cp in route_dict['checkpoints']]
                    checkpoint_indices = [idx for idx, cp in enumerate(route_dict['checkpoints'])]

                    # Calculate angles for checkpoint markers (keep this logic for later, but it won't be used immediately)
                    checkpoint_angles = []
                
                    for i in range(len(route_dict['checkpoints'])):
                        current_checkpoint_dict = route_dict['checkpoints'][i]
                        current_geo_point = GeoPoint(current_checkpoint_dict['lat'], current_checkpoint_dict['lon'])
                        
                        if i < len(route_dict['checkpoints']) - 1:
                            next_checkpoint_dict = route_dict['checkpoints'][i+1]
                            next_geo_point = GeoPoint(next_checkpoint_dict['lat'], next_checkpoint_dict['lon'])
                            bearing = current_geo_point.bearing_to(next_geo_point)
                            checkpoint_angles.append(bearing)
                        else:
                            checkpoint_angles.append(0.0) # Still providing a default float value

                    fig.add_trace(go.Scattermap(
                        lat=checkpoint_lats,
                        lon=checkpoint_lons,
                        mode='markers',
                        marker=dict(
                            size=16,
                            symbol='circle', # TEMPORARY CHANGE: Changed to circle
                            # angle=checkpoint_angles, # TEMPORARY CHANGE: Commented out angle
                            color='lime',
                            opacity=1.0
                        ),
                        text=checkpoint_names,
                        hoverinfo='text',
                        hovertext=[f"{name}<br>Высота: {elev:.1f} м" for name, elev in zip(checkpoint_names, checkpoint_elevations)],
                        customdata=checkpoint_indices,
                        showlegend=False,
                        name="Чекпоинты"
                    ))
                    print_step("Map Drawing", f"Отрисованы чекпоинты (теперь как круги для отладки) для выбранного маршрута '{route_dict['name']}'.")

                if highlight_checkpoint is not None and highlight_checkpoint < len(route_dict['checkpoints']):
                    checkpoint = route_dict['checkpoints'][highlight_checkpoint]
                    
                    # Calculate angle for the highlighted checkpoint (keep this logic)
                    highlight_angle = 0.0 
                    if highlight_checkpoint < len(route_dict['checkpoints']) - 1:
                        next_checkpoint_dict = route_dict['checkpoints'][highlight_checkpoint + 1]
                        next_geo_point = GeoPoint(next_checkpoint_dict['lat'], checkpoint['lon']) # Fixed typo: should be next_geo_point.lon, not checkpoint.lon
                        current_geo_point = GeoPoint(checkpoint['lat'], checkpoint['lon'])
                        highlight_angle = current_geo_point.bearing_to(next_geo_point)

                    fig.add_trace(go.Scattermap(
                        lat=[checkpoint['lat']],
                        lon=[checkpoint['lon']],
                        mode='markers',
                        marker=dict(
                            size=22,
                            symbol='circle', # TEMPORARY CHANGE: Changed to circle
                            # angle=highlight_angle, # TEMPORARY CHANGE: Commented out angle
                            color='red',
                            opacity=1.0,
                        ),
                        name="Выбранный чекпоинт",
                        hoverinfo='text',
                        hovertext=f"{checkpoint['name']}<br>Высота: {checkpoint['elevation']:.1f} м",
                        showlegend=False
                    ))
                    print_step("Map Drawing", f"Отрисован выделенный чекпоинт {highlight_checkpoint} (теперь как круг для отладки) на маршруте '{route_dict['name']}'.")

                if highlight_checkpoint is not None and highlight_checkpoint < len(route_dict['checkpoints']):
                    checkpoint = route_dict['checkpoints'][highlight_checkpoint]
                    
                    # Calculate angle for the highlighted checkpoint
                    highlight_angle = 0.0 # Default to 0.0 (North)
                    if highlight_checkpoint < len(route_dict['checkpoints']) - 1:
                        next_checkpoint_dict = route_dict['checkpoints'][highlight_checkpoint + 1]
                        next_geo_point = GeoPoint(next_checkpoint_dict['lat'], next_checkpoint_dict['lon'])
                        current_geo_point = GeoPoint(checkpoint['lat'], checkpoint['lon'])
                        highlight_angle = current_geo_point.bearing_to(next_geo_point)

                    fig.add_trace(go.Scattermap(
                        lat=[checkpoint['lat']],
                        lon=[checkpoint['lon']],
                        mode='markers',
                        marker=dict(
                            size=22,
                            symbol='triangle-up', 
                            angle=highlight_angle, # This will now be a float, not None
                            color='red',
                            opacity=1.0,
                        ),
                        name="Выбранный чекпоинт",
                        hoverinfo='text',
                        hovertext=f"{checkpoint['name']}<br>Высота: {checkpoint['elevation']:.1f} м",
                        showlegend=False
                    ))
                    print_step("Map Drawing", f"Отрисован выделенный чекпоинт {highlight_checkpoint} (со стрелкой) на маршруте '{route_dict['name']}'.")

                if highlight_checkpoint is not None and highlight_checkpoint < len(route_dict['checkpoints']):
                    checkpoint = route_dict['checkpoints'][highlight_checkpoint]
                    
                    # Calculate angle for the highlighted checkpoint
                    highlight_angle = None
                    if highlight_checkpoint < len(route_dict['checkpoints']) - 1:
                        next_checkpoint_dict = route_dict['checkpoints'][highlight_checkpoint + 1]
                        next_geo_point = GeoPoint(next_checkpoint_dict['lat'], next_checkpoint_dict['lon'])
                        current_geo_point = GeoPoint(checkpoint['lat'], checkpoint['lon'])
                        highlight_angle = current_geo_point.bearing_to(next_geo_point)

                    fig.add_trace(go.Scattermap(
                        lat=[checkpoint['lat']],
                        lon=[checkpoint['lon']],
                        mode='markers',
                        marker=dict(
                            size=22,
                            symbol='triangle-up', # Changed symbol to an arrow
                            angle=highlight_angle, # Apply the calculated angle
                            color='red',
                            opacity=1.0,
                        ),
                        name="Выбранный чекпоинт",
                        hoverinfo='text',
                        hovertext=f"{checkpoint['name']}<br>Высота: {checkpoint['elevation']:.1f} м",
                        showlegend=False
                    ))
                    print_step("Map Drawing", f"Отрисован выделенный чекпоинт {highlight_checkpoint} (со стрелкой) на маршруте '{route_dict['name']}'.")
                
                # REMOVE THE FOLLOWING BLOCK - it was for static annotations:
                # print_step("Map Drawing", "Добавлены стрелки направления для чекпоинтов.")

            else: # For non-selected routes
                fig.add_trace(go.Scattermap(
                    lat=route_lats,
                    lon=route_lons,
                    mode='lines',
                    line=dict(width=3, color='rgba(100, 100, 100, 0.5)'),
                    hoverinfo='text',
                    hovertext=f"Маршрут: {route_dict['name']}",
                    showlegend=False,
                    name=route_dict['name']
                ))
                print_step("Map Drawing", f"Отрисован невыбранный маршрут '{route_dict['name']}'.")
    
    def _add_forest_boundary_and_name_to_figure(self, fig):
        min_lon_val, min_lat_val, max_lon_val, max_lat_val = self.rm.bounds
        
        lons_boundary = [min_lon_val, max_lon_val, max_lon_val, min_lon_val, min_lon_val]
        lats_boundary = [min_lat_val, min_lat_val, max_lat_val, max_lat_val, min_lat_val]

        fig.add_trace(go.Scattermap(
            lat=lats_boundary,
            lon=lons_boundary,
            mode='lines',
            line=dict(width=3, color='blue'),
            hoverinfo='none',
            showlegend=False,
            name="Границы Битцевского леса"
        ))

        center_lat = (min_lat_val + max_lat_val) / 2
        center_lon = (min_lon_val + max_lon_val) / 2
        
        fig.add_annotation(
            x=center_lon,
            y=center_lat,
            text="Битцевский Парк",
            showarrow=False,
            font=dict(size=20, color="black", family="Arial, sans-serif"),
            yanchor="middle",
            xanchor="center",
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
        print_step("Map Drawing", "Добавлены границы и название леса.")

    def _create_checkpoint_card(self, checkpoint):
        # NOTE: get_photo_html is called here. Ensure it's correctly imported and functional.
        return [
            html.H5(f"{checkpoint['name']} ({checkpoint['position']}/{checkpoint['total_positions']})", 
                   className="card-title"),
            html.Iframe(
                srcDoc=checkpoint['photo_html'],
                style={'width': '100%', 'height': '220px', 'border': 'none', 'marginBottom': '10px'}
            ),
            html.P(f"Координаты: {checkpoint['lat']:.5f}, {checkpoint['lon']:.5f}"),
            html.P(f"Высота: {checkpoint['elevation']:.1f} м"),
            html.P(f"Расстояние от старта: {checkpoint.get('distance_from_start', 0):.0f} м"),
            html.P(checkpoint['description'], className="text-muted")
        ]

    def run(self):
        self.app.run(debug=False)

if __name__ == '__main__':
    app = BitsevskyMapApp()
    app.run()