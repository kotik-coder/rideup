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
        """Добавляет веломаршрут с цветовой интерполяцией по высоте"""
        try:
            points = [[float(p[0]), float(p[1])] for p in points]
            elevations = [float(e) for e in elevations]
            
            if len(points) != len(elevations):
                raise ValueError("Количество точек и высот не совпадает")

            # Создаем плавный маршрут
            smooth_route, route_elevations = self._create_smooth_route(points, elevations)
            
            # Добавляем цветовую карту
            colormap = self._create_colormap(elevations)
            
            # Отрисовываем маршрут
            self._draw_route(map_obj, smooth_route, route_elevations, colormap)
            
            # Добавляем стрелки направления
            self._add_direction_arrows(map_obj, smooth_route)
            
            # Добавляем маркеры с информацией
            self._add_info_markers(map_obj, points, names, elevations, descriptions, colormap)

        except Exception as e:
            print(f"Ошибка при построении маршрута: {str(e)}")

    def _create_colormap(self, elevations):
        """Создание цветовой карты на основе высот"""
        vmin, vmax = min(elevations), max(elevations)
        if vmin == vmax:
            vmin, vmax = vmin-10, vmax+10
            
        return cm.LinearColormap(
            ['#00aa00', '#ffff00', '#ff0000'],  # Зеленый-желтый-красный
            vmin=vmin,
            vmax=vmax
        )

    def _create_smooth_route(self, points, elevations):
        """Создает плавный маршрут с интерполяцией высот"""
        # Генерация контрольных точек
        control_points = []
        for i in range(1, len(points)-1):
            dx = (points[i+1][0] - points[i-1][0]) * 0.25
            dy = (points[i+1][1] - points[i-1][1]) * 0.25
            control_points.append((
                [points[i][0] - dx, points[i][1] - dy],
                [points[i][0] + dx, points[i][1] + dy]
            ))

        # Построение плавного маршрута
        smooth_route = []
        route_elevations = []
        
        # Первый сегмент
        if len(points) > 1:
            cp1 = [
                points[0][0] + (points[1][0]-points[0][0])*0.3,
                points[0][1] + (points[1][1]-points[0][1])*0.3
            ]
            cp2 = control_points[0][0] if control_points else points[1]
            segment = self._bezier_curve(points[0], cp1, cp2, points[1])
            smooth_route.extend(segment)
            
            # Интерполяция высот для первого сегмента
            for t in (i/len(segment) for i in range(len(segment))):
                elev = elevations[0] * (1-t) + elevations[1] * t
                route_elevations.append(elev)

        # Средние сегменты
        for i in range(1, len(points)-1):
            cp1 = control_points[i-1][1]
            cp2 = control_points[i][0] if i < len(control_points) else points[i+1]
            segment = self._bezier_curve(points[i], cp1, cp2, points[i+1])[1:]
            smooth_route.extend(segment)
            
            # Интерполяция высот для средних сегментов
            for t in (i/len(segment) for i in range(len(segment))):
                elev = elevations[i] * (1-t) + elevations[i+1] * t
                route_elevations.append(elev)

        return smooth_route, route_elevations

    def _draw_route(self, map_obj, smooth_route, route_elevations, colormap):
        # 6. Отрисовка маршрута с цветовой интерполяцией
        for i in range(len(smooth_route)-1):
            folium.PolyLine(
                locations=[smooth_route[i], smooth_route[i+1]],
                color=colormap(route_elevations[i]),
                weight=6,
                opacity=0.9
            ).add_to(map_obj)


    def _add_direction_arrows(self, map_obj, smooth_route):
        arrow_style = """
        <svg height="20" width="20">
        <path d="M0,5 L10,15 L20,5 L10,10 Z" fill="#000000" stroke="#ffffff" stroke-width="1"/>
        </svg>
        """
        
        for i in range(10, len(smooth_route)-10, max(1, len(smooth_route)//15)):
            p1 = smooth_route[i]
            p2 = smooth_route[i+5]
            angle = math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]) + math.pi)
            
            folium.Marker(
                location=smooth_route[i],
                icon=folium.DivIcon(
                    icon_size=(20,20),
                    icon_anchor=(10,10),
                    html=f'<div style="transform: rotate({angle}deg)">{arrow_style}</div>'
                ),
                z_index_offset=1000
            ).add_to(map_obj)

    def _add_info_markers(self, map_obj, points, names, elevations, descriptions, colormap):
        """Добавляет маркеры с фото и источником"""
        from media_helpers import get_photo_html
        for i, (point, name, elevation, desc) in enumerate(zip(points, names, elevations, descriptions)):
            delta = elevations[i] - elevations[i-1] if i > 0 else 0
            photo_html = get_photo_html(point[0], point[1])

            popup_content = f"""
            <div style="width:260px;font-family:Arial,sans-serif">
                <h4 style="margin:0;color:#333;border-bottom:1px solid #eee;padding-bottom:5px">
                    {i+1}. {name}
                </h4>
                {photo_html}
                <p style="color:#666;font-size:0.9em;margin:5px 0">{desc}</p>
                <div style="background:{colormap(elevation)};height:4px;margin:5px 0"></div>
                <p style="margin:3px 0"><b>Высота:</b> {elevation:.1f} м</p>
                <p style="margin:3px 0">
                    <b>Перепад:</b> <span style="color:{"#ff0000" if delta > 0 else "#00aa00"}">
                    {delta:+.1f} м</span>
                </p>
            </div>
            """
            
            folium.Marker(
                location=point,
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(
                    color='red' if i == 0 else 'darkgreen' if i == len(points)-1 else 'blue',
                    icon='flag' if i in (0, len(points)-1) else 'info-sign',
                    prefix='fa'
                )
            ).add_to(map_obj)
    
    def _bezier_curve(self, p0, cp1, cp2, p3, num_points=30):
        """Генерирует точки кривой Безье"""
        curve = []
        for t in (i/num_points for i in range(num_points+1)):
            lat = (1-t)**3*p0[0] + 3*(1-t)**2*t*cp1[0] + 3*(1-t)*t**2*cp2[0] + t**3*p3[0]
            lon = (1-t)**3*p0[1] + 3*(1-t)**2*t*cp1[1] + 3*(1-t)*t**2*cp2[1] + t**3*p3[1]
            curve.append([lat, lon])
        return curve

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