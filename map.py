import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import folium
import osmnx as ox
import branca.colormap as cm
from map_helpers import *

class BitsevskyMapWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Битцевский лес - Веломаршрут")
        self.setGeometry(100, 100, 800, 600)
        self.browser = QWebEngineView()
        self.setCentralWidget(self.browser)
        self.create_map_with_route()
    
    def create_map_with_route(self):
        """Создание карты с маршрутом"""
        try:
            # Получение границ леса
            forest = ox.geocode_to_gdf('Битцевский лес, Москва')
            forest_polygon = forest.geometry.iloc[0]
            bounds = [float(x) for x in forest.total_bounds]  # Преобразование в float
            
            # Создание базовой карты
            m = folium.Map(
                location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2],
                zoom_start=14,
                tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
                attr='OpenTopoMap',
                control_scale=True
            )

            # Генерация маршрута
            route_points, point_names, elevations, descriptions = self.generate_route(forest_polygon, bounds)
            
            # Добавление элементов
            self.add_gray_mask(m, bounds)
            self.add_bike_route(m, route_points, point_names, elevations, descriptions)
            
            # Сохранение и отображение
            map_path = os.path.abspath('bitsevsky_map.html')
            m.save(map_path)
            self.browser.setUrl(QUrl.fromLocalFile(map_path))
            
        except Exception as e:
            print(f"Ошибка при создании карты: {str(e)}", file=sys.stderr)

    def get_boundary_near_point(self, polygon, bounds, reference_point, max_distance=500):
        """Находит точку на границе не далее max_distance от reference_point внутри полигона"""
        ref_lat, ref_lon = float(reference_point[0]), float(reference_point[1])
        
        for _ in range(100):  # Максимум 100 попыток
            # Генерируем точку в пределах расстояния
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(100, float(max_distance))
            delta_lat = distance * math.cos(angle) / 111320
            delta_lon = distance * math.sin(angle) / (111320 * math.cos(math.radians(ref_lat)))
            
            candidate_lat = ref_lat + delta_lat
            candidate_lon = ref_lon + delta_lon
            candidate_point = Point(float(candidate_lon), float(candidate_lat))
            
            # Проверяем, что точка:
            # 1. Находится внутри границ леса
            # 2. Находится на границе или близко к ней
            if (polygon.contains(candidate_point) and 
                polygon.boundary.distance(candidate_point) < 0.0002):  # ~20 метров до границы
                return (float(candidate_lat), float(candidate_lon))
        
        # Если не нашли подходящую точку - возвращаем любую граничную внутри расстояния
        return get_boundary_point(polygon, bounds)

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

    def add_gray_mask(self, map_obj, bounds):
        """Добавляет серую маску на всю карту за пределами прямоугольника, обрамляющего лес"""
        try:
            # Определяем очень большие границы (весь возможный охват карты)
            WORLD_BOUNDS = [
                [bounds[1] - 10, bounds[0] - 10],  # SW (Юго-запад)
                [bounds[3] + 10, bounds[2] + 10]    # NE (Северо-восток)
            ]
            
            # Создаем полигон всей карты с "дыркой" в форме прямоугольника леса
            outer_polygon = [
                [WORLD_BOUNDS[0][0], WORLD_BOUNDS[0][1]],  # SW
                [WORLD_BOUNDS[0][0], WORLD_BOUNDS[1][1]],  # SE
                [WORLD_BOUNDS[1][0], WORLD_BOUNDS[1][1]],  # NE
                [WORLD_BOUNDS[1][0], WORLD_BOUNDS[0][1]]   # NW
            ]
            
            inner_polygon = [
                [bounds[1], bounds[0]],  # SW
                [bounds[1], bounds[2]],  # SE
                [bounds[3], bounds[2]],  # NE
                [bounds[3], bounds[0]]   # NW
            ]
            
            # Создаем многоугольник с отверстием
            folium.Polygon(
                locations=[outer_polygon, inner_polygon],
                color='none',
                fill=True,
                fill_color='gray',
                fill_opacity=0.7,
                interactive=False
            ).add_to(map_obj)
            
        except Exception as e:
            print(f"Ошибка при создании маски: {e}")

    def add_bike_route(self, map_obj, points, names, elevations, descriptions):
        """Добавляет веломаршрут с плавными кривыми и стрелками направления"""
        try:
            # 1. Подготовка данных
            points = [[float(p[0]), float(p[1])] for p in points]
            elevations = [float(e) for e in elevations]
            
            # 2. Функция Catmull-Rom сплайна
            def catmull_rom(p0, p1, p2, p3, num_points=25):
                """Генерирует точки сплайна между p1 и p2"""
                spline = []
                for t in (i/num_points for i in range(num_points+1)):
                    t2 = t * t
                    t3 = t2 * t
                    
                    lat = 0.5 * ((2 * p1[0]) + 
                                (-p0[0] + p2[0]) * t + 
                                (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 + 
                                (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)
                    
                    lon = 0.5 * ((2 * p1[1]) + 
                                (-p0[1] + p2[1]) * t + 
                                (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 + 
                                (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)
                    
                    spline.append([lat, lon])
                return spline

            # 3. Генерация плавного маршрута
            smooth_route = []
            for i in range(len(points)):
                if i == 0:
                    # Первый сегмент (дублируем первую точку)
                    segment = catmull_rom(points[0], points[0], points[1], points[2])
                elif i == len(points)-1:
                    # Последний сегмент (дублируем последнюю точку)
                    segment = catmull_rom(points[i-2], points[i-1], points[i], points[i])
                else:
                    # Средние сегменты
                    p0 = points[max(0, i-2)]
                    p1 = points[i-1]
                    p2 = points[i]
                    p3 = points[min(len(points)-1, i+1)]
                    segment = catmull_rom(p0, p1, p2, p3)
                
                smooth_route.extend(segment if i == 0 else segment[1:])

            # 4. Создание цветовой карты
            vmin, vmax = min(elevations), max(elevations)
            if vmin == vmax:
                vmin, vmax = vmin-10, vmax+10
                
            colormap = cm.LinearColormap(
                ['#00aa00', '#ffff00', '#ff0000'],  # Зеленый-Желтый-Красный
                vmin=vmin,
                vmax=vmax
            )

            # 5. Отрисовка маршрута с градиентом
            for i in range(len(smooth_route)-1):
                # Интерполяция высоты для текущего сегмента
                progress = i / len(smooth_route)
                elev_idx = int(progress * (len(elevations)-1))
                t = (progress * (len(elevations)-1)) % 1
                elev = elevations[elev_idx] * (1-t) + elevations[elev_idx+1] * t
                
                folium.PolyLine(
                    locations=[smooth_route[i], smooth_route[i+1]],
                    color=colormap(elev),
                    weight=8,
                    opacity=0.3
                ).add_to(map_obj)

                # 6. Добавление стрелок направления (каждые ~200 метров)
                if i % 15 == 0 and i < len(smooth_route)-5:
                    p1 = smooth_route[i]
                    p2 = smooth_route[i+3]
                    angle = math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0])) - 90
                    
                    folium.RegularPolygonMarker(
                        location=smooth_route[i+2],
                        number_of_sides=3,
                        radius=8,
                        rotation=angle,
                        color='#0055ff',
                        fill_color='#0055ff',
                        fill_opacity=1
                    ).add_to(map_obj)

            # 7. Добавление маркеров с информацией
            for i, (point, name, elevation, desc) in enumerate(zip(points, names, elevations, descriptions)):
                delta = elevations[i] - elevations[i-1] if i > 0 else 0
                
                popup_content = f"""
                <div style="width:260px;font-family:Arial,sans-serif">
                    <h4 style="margin:0;color:#333;border-bottom:1px solid #eee;padding-bottom:5px">
                        {i+1}. {name}
                    </h4>
                    <p style="color:#666;font-size:0.9em;margin:5px 0">{desc}</p>
                    <div style="background:{colormap(elevation)};height:4px;margin:5px 0"></div>
                    <p style="margin:3px 0"><b>Высота:</b> {elevation:.1f} м</p>
                    <p style="margin:3px 0">
                        <b>Перепад:</b> <span style="color:{"#ff0000" if delta > 0 else "#00aa00"}">
                        {delta:+.1f} м</span>
                    </p>
                </div>
                """
                
                icon_color = '#ff0000' if i == 0 else '#00aa00' if i == len(points)-1 else '#0055ff'
                
                folium.Marker(
                    location=point,
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(
                        color=icon_color,
                        icon='flag' if i in (0, len(points)-1) else 'circle',
                        prefix='fa'
                    )
                ).add_to(map_obj)

        except Exception as e:
            print(f"Ошибка при построении маршрута: {str(e)}")

    def _get_marker_icon(self, index, elevation, total_points):
        """Упрощенный метод создания иконки без цветовой карты"""
        if index == 0:
            return folium.Icon(color='red', icon='flag')
        elif index == total_points - 1:
            return folium.Icon(color='darkgreen', icon='flag')
        else:
            return folium.Icon(color='blue', icon='info-sign')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BitsevskyMapWindow()
    window.show()
    sys.exit(app.exec_())