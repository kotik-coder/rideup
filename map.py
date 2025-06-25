import sys
import os
import random
import math
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import folium
import osmnx as ox
from shapely.geometry import Point, LineString
import branca.colormap as cm

class BitsevskyMapWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Битцевский лес - Веломаршрут")
        self.setGeometry(100, 100, 800, 600)
        self.browser = QWebEngineView()
        self.setCentralWidget(self.browser)
        self.create_map_with_route()
    
    def create_map_with_route(self):
        try:
            forest = ox.geocode_to_gdf('Битцевский лес, Москва')
            forest_polygon = forest.geometry.iloc[0]
            bounds = forest.total_bounds
            
            # Создаем базовую карту
            m = folium.Map(
                location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2],
                zoom_start=14,
                tiles='OpenStreetMap',
                control_scale=True
            )
            
            # Добавляем спутниковый слой
            satelite_layer = folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Спутник',
                overlay=False,
                control=False
            ).add_to(m)
            
            # Генерация маршрута
            route_points, point_names = self.generate_route(forest_polygon, bounds)
            
            # Добавление элементов на карту
            self.add_gray_mask(m, [
                [bounds[1], bounds[0]],
                [bounds[3], bounds[2]]
            ])
            self.add_bike_route(m, route_points, point_names)
            
            # Добавляем скрипт для переключения слоев
            self.add_layer_switcher_script(m)
            
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
            map_path = os.path.abspath('bitsevsky_map.html')
            m.save(map_path)
            self.browser.setUrl(QUrl.fromLocalFile(map_path))
            
        except Exception as e:
            print(f"Ошибка: {e}")
    
    def add_layer_switcher_script(self, map_obj):
        """Добавляет JavaScript для автоматического переключения слоев по масштабу"""
        script = """
        // Порог переключения на спутник (при увеличении)
        var SATELLITE_ZOOM_THRESHOLD = 16;
        var currentLayer = 'scheme';
        var sateliteLayer = L.tileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            { attribution: 'Esri', maxZoom: 19 }
        );
        
        function updateMapLayer() {
            if (!map) return;
            
            var zoom = map.getZoom();
            
            // Переключаем на спутник при большом масштабе
            if (zoom >= SATELLITE_ZOOM_THRESHOLD && currentLayer !== 'satelite') {
                map.eachLayer(function(layer) {
                    if (layer._url && layer._url.includes('openstreetmap')) {
                        map.removeLayer(layer);
                    }
                });
                sateliteLayer.addTo(map);
                currentLayer = 'satelite';
            } 
            // Возвращаем схему при уменьшении
            else if (zoom < SATELLITE_ZOOM_THRESHOLD && currentLayer !== 'scheme') {
                map.eachLayer(function(layer) {
                    if (layer._url && layer._url.includes('arcgisonline')) {
                        map.removeLayer(layer);
                    }
                });
                L.tileLayer(
                    'https://{s}.tile.openstreetmap.org/{z}/{y}/{x}.png',
                    { attribution: 'OpenStreetMap', maxZoom: 19 }
                ).addTo(map);
                currentLayer = 'scheme';
            }
        }
        
        // Проверяем при изменении масштаба
        map.on('zoomend', updateMapLayer);
        // Инициализация
        setTimeout(updateMapLayer, 500);
        """
        
        map_obj.get_root().html.add_child(folium.Element(f"""
        <script>
            {script}
        </script>
        """))
    
    def generate_route(self, polygon, bounds):
        """Генерация маршрута с учётом всех правил"""
        point_names = ["Старт", "Овраг", "Ручей", "Поляна", "Роща", "Финиш"]
        
        # 1. Стартовая точка на границе
        start_point = self.get_boundary_point(polygon, bounds)
        points = [start_point]
        
        # 2. Промежуточные точки
        total_length = 0
        for _ in range(4):  # 4 промежуточных точки
            next_point = self.generate_nearby_point(
                points[-1], 
                polygon, 
                max_distance=500,
                bounds=bounds
            )
            if next_point:
                points.append(next_point)
                total_length += self.calculate_distance(points[-2], points[-1])
                if total_length > 13000:  # Оставляем место для финиша
                    break
        
        # 3. Финишная точка на границе (не далее 500м от последней точки)
        end_point = self.get_boundary_near_point(polygon, bounds, points[-1])
        points.append(end_point)
        
        return points, point_names[:len(points)]
    
    def get_boundary_near_point(self, polygon, bounds, reference_point, max_distance=500):
        """Находит точку на границе не далее max_distance от reference_point"""
        for _ in range(100):
            # Генерируем точку в пределах расстояния
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(100, max_distance)
            delta_lat = distance * math.cos(angle) / 111320
            delta_lon = distance * math.sin(angle) / (111320 * math.cos(math.radians(reference_point[0])))
            
            candidate = Point(
                reference_point[1] + delta_lon,
                reference_point[0] + delta_lat
            )
            
            # Проверяем, что точка на границе
            if (bounds[0] <= candidate.x <= bounds[2] and 
                bounds[1] <= candidate.y <= bounds[3] and
                polygon.boundary.distance(candidate) < 0.0001):
                return (candidate.y, candidate.x)
        
        # Если не нашли - возвращаем любую граничную точку
        return self.get_boundary_point(polygon, bounds)
    
    def get_boundary_point(self, polygon, bounds):
        """Генерирует точку на границе полигона"""
        for _ in range(100):
            side = random.choice(['north', 'south', 'east', 'west'])
            if side == 'north':
                lat, lon = bounds[3], random.uniform(bounds[0], bounds[2])
            elif side == 'south':
                lat, lon = bounds[1], random.uniform(bounds[0], bounds[2])
            elif side == 'east':
                lat, lon = random.uniform(bounds[1], bounds[3]), bounds[2]
            else:
                lat, lon = random.uniform(bounds[1], bounds[3]), bounds[0]
            
            if polygon.boundary.distance(Point(lon, lat)) < 0.0001:
                return (lat, lon)
        return (bounds[1], bounds[0])
    
    def generate_nearby_point(self, reference_point, polygon, max_distance, bounds):
        """Генерирует точку не далее max_distance от reference_point"""
        for _ in range(100):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(100, max_distance)
            delta_lat = distance * math.cos(angle) / 111320
            delta_lon = distance * math.sin(angle) / (111320 * math.cos(math.radians(reference_point[0])))
            
            new_lat = reference_point[0] + delta_lat
            new_lon = reference_point[1] + delta_lon
            
            if (bounds[1] <= new_lat <= bounds[3] and 
                bounds[0] <= new_lon <= bounds[2] and
                polygon.contains(Point(new_lon, new_lat))):
                return (new_lat, new_lon)
        return None
    
    def calculate_distance(self, p1, p2):
        """Вычисляет расстояние между точками в метрах"""
        return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) * 111320
    
    def add_gray_mask(self, map_obj, bounds):
        """Добавляет серую маску вокруг области"""
        outer = [
            [bounds[0][0]-1, bounds[0][1]-1],
            [bounds[0][0]-1, bounds[1][1]+1],
            [bounds[1][0]+1, bounds[1][1]+1],
            [bounds[1][0]+1, bounds[0][1]-1]
        ]
        inner = [
            bounds[0],
            [bounds[0][0], bounds[1][1]],
            bounds[1],
            [bounds[1][0], bounds[0][1]]
        ]
        folium.Polygon(
            locations=[outer, inner],
            color='none',
            fill=True,
            fill_color='gray',
            fill_opacity=0.5,
            interactive=False
        ).add_to(map_obj)
    
    def add_bike_route(self, map_obj, points, names):
        """Добавляет маршрут с маркерами"""
        # Градиентная линия
        folium.ColorLine(
            positions=points,
            colors=range(len(points)),
            colormap=cm.LinearColormap(['#006400', '#32CD32']),
            weight=5,
            opacity=0.8
        ).add_to(map_obj)
        
        # Маркеры с подписями
        for i, (point, name) in enumerate(zip(points, names)):
            folium.Marker(
                location=point,
                icon=folium.Icon(
                    color='red' if i == 0 else 'darkgreen' if i == len(points)-1 else 'blue',
                    icon='flag' if i in (0, len(points)-1) else 'circle'
                )
            ).add_to(map_obj)
            
            folium.Marker(
                location=[point[0]+0.0003, point[1]+0.0003],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:14px;font-weight:bold;color:#333;text-shadow:-1px 0 white, 0 1px white, 1px 0 white, 0 -1px white;">{i+1}. {name}</div>'
                )
            ).add_to(map_obj)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BitsevskyMapWindow()
    window.show()
    sys.exit(app.exec_())