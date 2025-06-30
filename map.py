import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import Akima1DInterpolator
from gpx_loader import LocalGPXLoader
from route_manager import RouteManager
from route import GeoPoint
# import branca.colormap as cm # Уберите, если не используется для других целей Plotly
from datetime import datetime
from map_helpers import print_step
import json
from media_helpers import get_photo_html, get_landscape_photo # get_landscape_photo пока не используется напрямую здесь, но пригодится
from typing import List, Tuple, Dict # ИСПРАВЛЕНО: Добавлен импорт List, Tuple, Dict
from pathlib import Path # Необходимо для Path(path).name в _add_photo_markers

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
        
        # Corrected: Unpack bounds in [min_lon, min_lat, max_lon, max_lat] order
        min_lon_val, min_lat_val, max_lon_val, max_lat_val = self.rm.bounds # Correct unpacking
        
        center_lat = (min_lat_val + max_lat_val) / 2 # Use corrected lat values
        center_lon = (min_lon_val + max_lon_val) / 2 # Use corrected lon values

        # Calculate initial zoom based on the full forest bounds (using corrected lat/lon)
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
                for i, route in enumerate(self.rm.valid_routes):
                    route_dict = self._process_route(route)
                    route_data.append(route_dict)
                    options.append({'label': route.name, 'value': i})
            
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
                return fig

            route_data = json.loads(route_data_json)

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
            else:
                # If no route is selected, zoom to the entire forest area
                # Corrected: Unpack bounds in [min_lon, min_lat, max_lon, max_lat] order
                min_lon_val, min_lat_val, max_lon_val, max_lat_val = self.rm.bounds # Correct unpacking
                center_lat = (min_lat_val + max_lat_val) / 2 # Use corrected lat values
                center_lon = (min_lon_val + max_lon_val) / 2 # Use corrected lon values
                zoom = self._calculate_zoom([min_lat_val, max_lat_val], [min_lon_val, max_lon_val])
                fig.update_layout(
                    map_center={'lat': center_lat, 'lon': center_lon},
                    map_zoom=zoom
                )
                self._add_forest_boundary_and_name_to_figure(fig) #


            # Drawing all routes
            for i, route_dict in enumerate(route_data):
                # Always add non-selected routes first
                if i != route_index:
                    self._add_route_to_figure(fig, route_dict, is_selected=False)
            
            # Add the selected route last to ensure it's on top
            if route_index is not None and route_index < len(route_data):
                selected_route_dict = route_data[route_index]
                self._add_route_to_figure(
                    fig, 
                    selected_route_dict, 
                    is_selected=True,
                    highlight_checkpoint=checkpoint_index
                )
                
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
                return "Выберите маршрут", empty_fig, "Выберите чекпоинт на карте"
                
            route_data = json.loads(route_data_json)
            if selected_route_index >= len(route_data):
                return dash.no_update, dash.no_update, dash.no_update

            route = route_data[selected_route_index]
            
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
                fill='tozeroy', # Keep fill tozeroy if you want the area filled down to the axis
                name='Высота'
            ))

            # --- MODIFICATION START ---
            # Calculate min and max elevation from the profile
            elevations_in_profile = [x['elevation'] for x in route['elevation_profile']]
            if elevations_in_profile:
                min_elevation = min(elevations_in_profile)
                max_elevation = max(elevations_in_profile)
                
                # Add a small offset below the minimum elevation
                y_axis_min = min_elevation - (max_elevation - min_elevation) * 0.1 # 10% offset
                if y_axis_min < 0: # Ensure y_axis_min doesn't go below 0 if elevations are low
                    y_axis_min = 0 
            else:
                y_axis_min = 0 # Default if no elevations
            # --- MODIFICATION END ---
            
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
                yaxis=dict(range=[y_axis_min, max_elevation + (max_elevation - min_elevation) * 0.1]) # Добавляем 10% от диапазона высот как верхний отступ
            )
            
            if checkpoint_index is not None and checkpoint_index < len(route['checkpoints']):
                checkpoint_info = self._create_checkpoint_card(route['checkpoints'][checkpoint_index])
            else:
                checkpoint_info = "Выберите чекпоинт на карте"
            
            return info_content, elevation_fig, checkpoint_info

    def _calculate_zoom(self, lats, lons):
        """
        --- Исправлено: Более надежный расчет зума ---
        Вычисляет уровень масштабирования на основе географического охвата.
        """
        if not lats or not lons:
            return 12 

        lat_span = max(lats) - min(lats)
        lon_span = max(lons) - min(lons)
        
        # Обработка случая с одной точкой или нулевым охватом
        if lat_span == 0 and lon_span == 0:
            return 15

        # These constants are empirical and might need fine-tuning for your specific map
        # A rough conversion from span in degrees to a zoom level
        # A smaller span means a higher zoom level
        zoom_lat = 9.5 - np.log2(lat_span + 1e-6)
        zoom_lon = 9.5 - np.log2(lon_span + 1e-6)

        return min(zoom_lat, zoom_lon, 18) # Cap max zoom to 18

    def _add_route_to_figure(self, fig, route_dict, is_selected=False, highlight_checkpoint=None):
        route_lats = [p[0] for p in route_dict['smooth_points']]
        route_lons = [p[1] for p in route_dict['smooth_points']]
        route_elevations = route_dict['elevation_profile'] # Получаем данные о высоте

        if not route_lats or not route_lons or not route_elevations:
            return

        # Если маршрут выбран, применяем цветовую схему по высоте
        if is_selected:
            # 1. Определяем минимальную, максимальную и стартовую высоту
            elevations_only = [ep['elevation'] for ep in route_elevations]
            if not elevations_only:
                return

            min_elev = min(elevations_only)
            max_elev = max(elevations_only)
            start_elev = elevations_only[0] if elevations_only else 0

            # 2. Определяем Plotly colorscale (как числовые диапазоны)
            if min_elev == max_elev: # Избегаем деления на ноль для плоских маршрутов
                plotly_colorscale = [[0, 'green'], [1, 'green']]
                cmin_val = min_elev - 1 if min_elev > 0 else 0
                cmax_val = max_elev + 1
            else:
                norm_start_elev = (start_elev - min_elev) / (max_elev - min_elev)
                plotly_colorscale = [
                    [0, 'blue'],               # Минимальная высота - синий
                    [norm_start_elev, 'green'],# Высота старта - зеленый
                    [1, 'red']                 # Максимальная высота - красный
                ]
                plotly_colorscale.sort(key=lambda x: x[0]) # Убедимся, что точки отсортированы
                cmin_val = min_elev
                cmax_val = max_elev

            # Добавляем маршрут с невидимой линией и цветными маркерами
            fig.add_trace(go.Scattermap(
                lat=route_lats,
                lon=route_lons,
                mode='lines+markers', # Используем линии и маркеры
                line=dict(width=6, color='rgba(0,0,0,0)'), # Делаем линию полностью прозрачной
                marker=dict(
                    size=12, # Увеличиваем размер маркеров, чтобы они сливались и создавали эффект линии
                    color=elevations_only, # Передаем числовые значения высоты для раскрашивания маркеров
                    colorscale=plotly_colorscale, # Цветоваядкала для маркеров
                    cmin=cmin_val,
                    cmax=cmax_val,
                    colorbar=dict( # Цветовая шкала для легенды (связана с высотой маркеров)
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

            if route_dict['checkpoints']:
                checkpoint_lats = [cp['lat'] for cp in route_dict['checkpoints']]
                checkpoint_lons = [cp['lon'] for cp in route_dict['checkpoints']]
                checkpoint_names = [cp['name'] for cp in route_dict['checkpoints']]
                checkpoint_elevations = [cp['elevation'] for cp in route_dict['checkpoints']]
                checkpoint_indices = [idx for idx, cp in enumerate(route_dict['checkpoints'])] # Используем 0-базовый индекс для customdata

                fig.add_trace(go.Scattermap(
                    lat=checkpoint_lats,
                    lon=checkpoint_lons,
                    mode='markers',
                    marker=dict(
                        size=12,
                        symbol='circle',
                        color='blue',
                        opacity=0.5
                    ),
                    text=checkpoint_names, # Для отображения названия при наведении
                    hoverinfo='text',
                    hovertext=[f"{name}<br>Высота: {elev:.1f} м" for name, elev in zip(checkpoint_names, checkpoint_elevations)],
                    customdata=checkpoint_indices, # Для передачи индекса чекпоинта по клику
                    showlegend=False,
                    name="Чекпоинты"
                ))

            # Выделенный чекпоинт (этот блок уже есть в вашем файле)
            if highlight_checkpoint is not None and highlight_checkpoint < len(route_dict['checkpoints']):
                checkpoint = route_dict['checkpoints'][highlight_checkpoint]
                fig.add_trace(go.Scattermap(
                    lat=[checkpoint['lat']],
                    lon=[checkpoint['lon']],
                    mode='markers',
                    marker=dict(
                        size=16, # Больше для выделенного
                        symbol='circle',
                        color='yellow',
                        opacity=0.5,
                    ),
                    name="Выбранный чекпоинт",
                    hoverinfo='text',
                    hovertext=f"{checkpoint['name']}<br>Высота: {checkpoint['elevation']:.1f} м",
                    showlegend=False
                ))

        else: # Если маршрут не выбран, отображаем его как обычную серую линию
            fig.add_trace(go.Scattermap(
                lat=route_lats,
                lon=route_lons,
                mode='lines',
                line=dict(width=3, color='rgba(100, 100, 100, 0.5)'), # Полупрозрачная серая линия
                hoverinfo='text',
                hovertext=f"Маршрут: {route_dict['name']}",
                showlegend=False,
                name=route_dict['name']
            ))
    
    def _add_forest_boundary_and_name_to_figure(self, fig):
        min_lon_val, min_lat_val, max_lon_val, max_lat_val = self.rm.bounds #
        
        # Coordinates for the rectangular boundary of the forest
        lons_boundary = [min_lon_val, max_lon_val, max_lon_val, min_lon_val, min_lon_val] #
        lats_boundary = [min_lat_val, min_lat_val, max_lat_val, max_lat_val, min_lat_val] #

        fig.add_trace(go.Scattermap( #
            lat=lats_boundary, #
            lon=lons_boundary, #
            mode='lines', #
            line=dict(width=3, color='blue'), # Distinct blue border
            hoverinfo='none', #
            showlegend=False, #
            name="Границы Битцевского леса" #
        ))

        # Add annotation for the park's name
        center_lat = (min_lat_val + max_lat_val) / 2 #
        center_lon = (min_lon_val + max_lon_val) / 2 #
        
        fig.add_annotation( #
            x=center_lon, #
            y=center_lat, #
            text="Битцевский Парк", #
            showarrow=False, #
            font=dict(size=20, color="black", family="Arial, sans-serif"), #
            yanchor="middle", #
            xanchor="center", #
            bgcolor="rgba(255, 255, 255, 0.7)", # Slightly transparent background
            bordercolor="black", #
            borderwidth=1, #
            borderpad=4 #
        )


    def _process_route(self, route):
        smooth_points, smooth_elevations = self._create_smooth_route(route)
        checkpoints = self._get_checkpoints(route, smooth_points, smooth_elevations)
        
        if smooth_points and checkpoints:
            total_dist_so_far = 0
            current_smooth_idx = 0
            for cp in checkpoints:
                target_smooth_idx = cp['point_index']
                # Accumulate distance from current_smooth_idx to target_smooth_idx
                for k in range(current_smooth_idx, min(target_smooth_idx, len(smooth_points) - 1)):
                    p1 = GeoPoint(*smooth_points[k])
                    p2 = GeoPoint(*smooth_points[k+1])
                    total_dist_so_far += p1.distance_to(p2)
                cp['distance_from_start'] = total_dist_so_far
                current_smooth_idx = target_smooth_idx

        segments = self._calculate_segments(checkpoints, smooth_elevations)
        elevation_profile = self._create_elevation_profile(smooth_points, smooth_elevations)
        
        return {
            'name': route.name,
            'checkpoints': checkpoints,
            'segments': segments,
            'elevation_profile': elevation_profile,
            'raw_points': [(p.lat, p.lon) for p in route.points],
            'raw_elevations': route.elevations,
            'smooth_points': smooth_points
        }

    def _create_elevation_profile(self, points, elevations):
        profile = []
        total_distance = 0
        if not points: return []
        
        profile.append({'distance': 0, 'elevation': elevations[0]})
        for i in range(1, len(points)):
            p1 = GeoPoint(*points[i-1])
            p2 = GeoPoint(*points[i])
            total_distance += p1.distance_to(p2)
            profile.append({'distance': total_distance, 'elevation': elevations[i]})
        return profile

    def _get_checkpoints(self, route, smooth_route, route_elevations):
        if len(smooth_route) < 2: return []

        distances = [0.0]
        for i in range(1, len(smooth_route)):
            p1 = GeoPoint(*smooth_route[i-1])
            p2 = GeoPoint(*smooth_route[i])
            distances.append(distances[-1] + p1.distance_to(p2))
        
        total_length = distances[-1]
        
        target_markers = min(20, max(5, int(total_length / 250))) if total_length > 0 else 2

        marker_indices = {0, len(smooth_route) - 1}
        if target_markers > 2 and total_length > 0:
            step_length = total_length / (target_markers - 1)
            for i in range(1, target_markers - 1):
                target_dist = i * step_length
                closest_idx = min(range(len(distances)), key=lambda x: abs(distances[x] - target_dist))
                marker_indices.add(closest_idx)
        
        sorted_indices = sorted(list(marker_indices))
        
        checkpoints = []
        for i, idx in enumerate(sorted_indices):
            point = smooth_route[idx]
            
            closest_original_idx = min(range(len(route.points)), key=lambda x: GeoPoint(route.points[x].lat, route.points[x].lon).distance_to(GeoPoint(*point)))
            
            point_name = f"Точка {i+1}"
            if i == 0: point_name = "Старт"
            elif i == len(sorted_indices) - 1: point_name = "Финиш"
            
            checkpoint = {
                'point_index': idx,
                'position': i + 1,
                'total_positions': len(sorted_indices),
                'lat': point[0],
                'lon': point[1],
                'elevation': route_elevations[idx],
                'name': point_name,
                'description': route.descriptions[closest_original_idx] if closest_original_idx < len(route.descriptions) else "",
                'photo_html': get_photo_html(point[0], point[1])
            }
            checkpoints.append(checkpoint)
        
        return checkpoints

    def _calculate_segments(self, checkpoints, elevations):
        segments = []
        for i in range(1, len(checkpoints)):
            start_cp = checkpoints[i-1]
            end_cp = checkpoints[i]
            
            start_idx = start_cp['point_index']
            end_idx = end_cp['point_index']
            
            segment_elevations = elevations[start_idx:end_idx+1]
            if not segment_elevations: continue

            segments.append({
                'distance': end_cp.get('distance_from_start', 0) - start_cp.get('distance_from_start', 0),
                'elevation_gain': max(0, max(segment_elevations) - start_cp['elevation']),
                'elevation_loss': max(0, start_cp['elevation'] - min(segment_elevations)),
                'net_elevation': end_cp['elevation'] - start_cp['elevation'],
                'start_checkpoint': start_cp['position'] - 1,
                'end_checkpoint': end_cp['position'] - 1
            })
        return segments

    def _create_checkpoint_card(self, checkpoint):
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
    
    def _create_smooth_route(self, route):
        if len(route.points) < 4:
            return [(p.lat, p.lon) for p in route.points], route.elevations
            
        points = np.array([(p.lat, p.lon) for p in route.points])
        elevations = np.array(route.elevations)
        
        distances = np.zeros(len(points))
        for i in range(1, len(points)):
            p1 = GeoPoint(points[i-1][0], points[i-1][1])
            p2 = GeoPoint(points[i][0], points[i][1])
            distances[i] = distances[i-1] + p1.distance_to(p2)
        
        if distances[-1] == 0:
            return [(p.lat, p.lon) for p in route.points], route.elevations

        t = distances / distances[-1]
        
        num_smooth_points = max(100, int(distances[-1] / 10))
        t_smooth = np.linspace(0, 1, num_smooth_points)
        
        try:
            spline_lat = Akima1DInterpolator(t, points[:, 0])
            spline_lon = Akima1DInterpolator(t, points[:, 1])
            spline_elev = Akima1DInterpolator(t, elevations)
            
            smooth_lats = spline_lat(t_smooth)
            smooth_lons = spline_lon(t_smooth)
            smooth_elevs = spline_elev(t_smooth)
            
            smooth_points = list(zip(smooth_lats, smooth_lons))
            return smooth_points, smooth_elevs.tolist()

        except Exception as e:
            print(f"Could not create spline, returning raw points. Error: {e}")
            return [(p.lat, p.lon) for p in route.points], route.elevations


    def run(self):
        self.app.run(debug=False)

if __name__ == '__main__':
    app = BitsevskyMapApp()
    app.run()