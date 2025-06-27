import osmnx as ox

from datetime import datetime

from pandas import DataFrame
import map_helpers
import folium
from map_helpers import * 
from gpx_loader import *

from typing import List
from dataclasses import dataclass

from PyQt5.QtCore import QUrl

class RouteManager():

    map          : folium.Map
    bounds       : List[float]
    valid_routes : List[Route]

    def __init__(self, geostring : str):
        self.map, self.bounds = self.create_blank_map(geostring)
        self.valid_routes     = self.load_valid_routes()

    def load_valid_routes(self):
        print_step("Маршруты", "Загружаю маршруты...")
        loader = LocalGPXLoader()
        valid_routes = [ r \
                            for r in loader.load_routes() \
                            if r.is_valid_route(self.bounds) ]
        print_step("Маршруты", f"Найдено {len(valid_routes)} валидных маршрутов")
        return valid_routes

    def create_blank_map(self, geostring: str):
        # Получение границ леса
        print_step(prefix="Маршруты", message=f"Получаю границы области {geostring}...")
        forest = ox.geocode_to_gdf(geostring)
        forest_polygon = forest.geometry.iloc[0]
        
        bounds = [float(x) for x in forest.total_bounds]
        cx = (bounds[1] + bounds[3])/2
        cy = (bounds[0] + bounds[2])/2
        
        # Создание базовой карты
        print_step("Маршруты", "Создаю базовую карту...")
        m = folium.Map(
            location=[cx, cy],
            zoom_start=14,
            tiles=None,
            control_scale=True
        )

        # 1. Добавляем серый фон для всей карты (внешняя область)
        folium.TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
            name='Grayscale Background',
            attr='CartoDB',
            opacity=0.7,
            control=False
        ).add_to(m)

        # 2. Создаем GeoJSON с полигоном леса
        if forest_polygon.geom_type == 'MultiPolygon':
            features = []
            for poly in forest_polygon.geoms:
                coords = [[y, x] for x, y in zip(poly.exterior.xy[0], poly.exterior.xy[1])]
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    },
                    "properties": {}
                })
            geojson_data = {"type": "FeatureCollection", "features": features}
        else:
            coords = [[y, x] for x, y in zip(forest_polygon.exterior.xy[0], forest_polygon.exterior.xy[1])]
            geojson_data = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    },
                    "properties": {}
                }]
            }

        # 3. Создаем FeatureGroup для детальной карты
        detailed_group = folium.FeatureGroup(name='Detailed Area').add_to(m)
        '''
        # Добавляем полигон как маску
        folium.GeoJson(
            geojson_data,
            style_function=lambda x: {
                "fillColor": "#ffffff",
                "fillOpacity": 1,
                "color": "none",
                "weight": 0
            }
        ).add_to(detailed_group)
        
        # Добавляем детальные тайлы в группу
        folium.TileLayer(
            tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
            name='Detailed Map',
            attr='OpenTopoMap',
            opacity=1.0,
            control=False
        ).add_to(detailed_group)
        '''

        # 4. Добавляем границы леса поверх всего
        folium.GeoJson(
            geojson_data,
            name='Forest Boundary',
            style_function=lambda x: {
                "color": "#2ca25f",
                "weight": 3,
                "fillOpacity": 0
            }
        ).add_to(m)

        # 5. Управление порядком слоев через CSS
        m.add_child(folium.Element("""
            <style>
                .leaflet-layer:nth-child(1) {  /* Серый фон */
                    filter: grayscale(100%) opacity(70%);
                    z-index: 100;
                }
                .leaflet-layer:nth-child(2) {  /* Детальная карта */
                    z-index: 200;
                }
                .leaflet-overlay-pane {  /* Границы */
                    z-index: 300;
                }
            </style>
        """))

        return m, bounds

    def save_map(self) -> QUrl:
        # Сохранение и отображение
        try: 
            print_step("Маршруты", "Сохраняю карту...")
            
            # Создаем папку для карт (абсолютный путь)
            maps_dir = Path(__file__).parent / "maps"
            maps_dir.mkdir(exist_ok=True)
            
            # Генерируем уникальное имя файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            map_filename = f"bitsevsky_map_{timestamp}.html"
            map_path = str(maps_dir / map_filename)
            
            # Сохраняем карту
            self.map.save(map_path)
            
            # Преобразуем путь для QUrl (абсолютный путь)
            absolute_path = Path(map_path).absolute()
            url = QUrl.fromLocalFile(str(absolute_path))
            
            print_step("Маршруты", f"Карта сохранена: {map_path}")
            print_step("Маршруты", f"Абсолютный путь: {absolute_path}")
            print_step("Маршруты", f"URL для загрузки: {url.toString()}")
            
            return url
        
        except Exception as e:
            print_step("Маршруты", f"Ошибка при создании карты: {str(e)}")
            raise    

    def generate_route(self, polygon, bounds):
        """Генерация маршрута с проверкой всех точек"""
        point_names = [
            "Старт (южный вход)",
            "Овраг у ручья", 
            "Сосновая роща",
            "Центральная поляна",
            "Дубовая аллея",
            "Финиш (северный выход)"
        ]
        
        points = []
        elevations = []
        descriptions = []
        
        # 1. Стартовая точка (обязательно на границе)
        start_point = get_boundary_point(polygon, bounds)
        points.append(start_point)
        elevations.append(safe_get_elevation(*start_point))
        descriptions.append("Начало маршрута у главного входа")
        
        # 2. Промежуточные точки (внутри леса)
        for i in range(4):
            next_point = None
            attempts = 0
            
            while attempts < 50:  # Максимум 50 попыток
                candidate = generate_nearby_point(points[-1], polygon, 500, bounds)
                if candidate and polygon.contains(Point(float(candidate[1]), float(candidate[0]))):
                    next_point = candidate
                    break
                attempts += 1
            
            if next_point:
                points.append(next_point)
                elevations.append(safe_get_elevation(*next_point))
                delta = elevations[-1] - elevations[-2] if i > 0 else 0
                descriptions.append(get_landscape_description(elevations[-1], delta))
        
        # 3. Финишная точка (на границе, не далее 500м от последней точки)
        end_point = None
        attempts = 0
        
        while attempts < 100:
            candidate = self.get_boundary_near_point(polygon, bounds, points[-1], 500)
            if (candidate and 
                polygon.boundary.distance(Point(float(candidate[1]), float(candidate[0]))) < 0.0002):
                end_point = candidate
                break
            attempts += 1
        
        if not end_point:
            end_point = get_boundary_point(polygon, bounds)
        
        points.append(end_point)
        elevations.append(safe_get_elevation(*end_point))
        descriptions.append("Конечная точка маршрута")
        
        return points, point_names, elevations, descriptions