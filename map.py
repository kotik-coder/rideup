import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl

import osmnx as ox
import branca.colormap as cm
import math
import random
from media_helpers import *
from shapely.geometry import Point

from map_helpers import *
from gpx_loader import LocalGPXLoader  # Импортируем наш загрузчик

import time
from datetime import datetime
from PyQt5.QtCore import QTimer

from route_manager import *

from scipy.interpolate import Akima1DInterpolator
import numpy as np

location_descriptor = 'Битцевский лес, Москва'

class BitsevskyMapWindow(QMainWindow):

    rm : RouteManager

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Битцевский лес - Веломаршрут")
        self.setGeometry(100, 100, 800, 600)
        self.browser = QWebEngineView()
        self.setCentralWidget(self.browser)
        self._print_header()
        rm = RouteManager("Битцевский лес, Москва")        
        # Загружаем в QWebEngineView
        self.populate_map_with_routes(rm)
        url = rm.save_map()
        self.browser.setUrl(url)

    def _print_header(self):
        """Выводит заголовок с информацией о запуске"""
        print("\n" + "="*50)
        print(f"Загрузка карты Битцевского леса")
        print(f"Время начала: {datetime.now().strftime('%H:%M:%S')}")
        print("="*50 + "\n")

    def populate_map_with_routes(self, rm : RouteManager):
        """Создание карты с маршрутом"""
        routes = rm.valid_routes
        m = rm.map
        
        # Добавление маршрутов на карту только если они есть
        if routes:
            print_step("Карта", f"Найдено {len(routes)} маршрутов")
            for route in routes:
                self.add_bike_route( m, route)
        else:
            print_step("Карта","Маршруты не найдены. Карта будет создана без маршрутов.")                            

    def add_bike_route(self, map : folium.Map, r : Route):
        """Добавляет веломаршрут с цветовой интерполяцией по высоте и оптимизированными чекпоинтами"""
        try:
            print_step("Карта","Начало построения маршрута...")

            # Создаем плавный маршрут
            print_step("Карта","Создание плавного маршрута...")
            smooth_route, route_elevations = self._create_smooth_route(r)
            
            # Добавляем цветовую карту
            print_step("Карта","Создание цветовой карты высот...")
            colormap = self._create_colormap(r.elevations)
            
            # Отрисовываем маршрут
            print_step("Карта","Отрисовка маршрута...")
            self._draw_route(map, smooth_route, route_elevations, colormap)            
            
            # Добавляем маркеры с информацией (оптимизированные чекпоинты)
            print_step("Карта","Добавление информационных маркеров...")
            r_dict = r.to_map_format()
            checkpoints = self._add_optimized_info_markers(map, r_dict['points'], 
                                                  r_dict['names'], 
                                                  r_dict['elevations'], 
                                                  r_dict['descriptions'], 
                                                  colormap, 
                                                  smooth_route)
            
            # Добавляем стрелки направления
            print_step("Карта","Добавление стрелок направления...")
            self._add_direction_arrows(map, smooth_route, checkpoints)

            print_step("Карта","Маршрут успешно добавлен на карту")

        except Exception as e:
            print_step("Карта", f"Ошибка при построении маршрута: {str(e)}")
            raise

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
    
    def _create_smooth_route(self, route : Route):
        points = route.points
        elevations = route.elevations

        # Преобразуем точки в массивы numpy
        lats = np.array([p[0] for p in points])
        lons = np.array([p[1] for p in points])
        elevs = np.array(elevations)

        # Параметр t (равномерно распределённый от 0 до 1)
        t = np.linspace(0, 1, num=len(points))

        # Создаём сплайны для широты, долготы и высоты
        spline_lat = Akima1DInterpolator(t, lats)
        spline_lon = Akima1DInterpolator(t, lons)
        spline_elev = Akima1DInterpolator(t, elevs)

        # Генерируем сглаженные точки (100 точек на маршрут)
        t_smooth = np.linspace(0, 1, num=100)
        smooth_lats = spline_lat(t_smooth)
        smooth_lons = spline_lon(t_smooth)
        smooth_elevs = spline_elev(t_smooth)

        # Объединяем в список кортежей (lat, lon)
        smooth_route = list(zip(smooth_lats, smooth_lons))
        route_elevations = smooth_elevs.tolist()

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

    def _add_direction_arrows(self, map_obj, smooth_route, checkpoints):
        """Добавляет стрелки направления между всеми парами чекпоинтов"""
        print_step("Карта", f"Добавление стрелок направления для маршрута из {len(smooth_route)} точек")

        # Сохраняем оригинальный стиль стрелки
        arrow_style = """
        <svg height="20" width="20">
        <path d="M0,5 L10,15 L20,5 L10,10 Z" fill="#000000" stroke="#ffffff" stroke-width="1"/>
        </svg>
        """

        # 1. Находим точные позиции чекпоинтов на сглаженном маршруте
        checkpoint_positions = []
        for checkpoint in checkpoints:
            closest_idx = min(
                range(len(smooth_route)),
                key=lambda i: checkpoint.distance_to(GeoPoint(*smooth_route[i]))
            )
            checkpoint_positions.append(closest_idx)
        
        # Сортируем позиции по порядку следования
        checkpoint_positions.sort()

        # 2. Добавляем ровно по одной стрелке между каждой парой чекпоинтов
        arrows_added = 0
        for i in range(len(checkpoint_positions)-1):
            start_idx = checkpoint_positions[i]
            end_idx = checkpoint_positions[i+1]
            
            # Вычисляем середину между чекпоинтами
            mid_idx = (start_idx + end_idx) // 2
            
            # Проверяем, чтобы не выйти за границы маршрута
            if mid_idx >= len(smooth_route)-5:
                continue
                
            # Вычисляем направление движения
            p1 = smooth_route[mid_idx]
            p2 = smooth_route[mid_idx+5]
            angle = math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]) + math.pi)

            # Добавляем стрелку
            folium.Marker(
                location=smooth_route[mid_idx],
                icon=folium.DivIcon(
                    icon_size=(20,20),
                    icon_anchor=(10,10),
                    html=f'<div style="transform: rotate({angle}deg)">{arrow_style}</div>'
                ),
                z_index_offset=1000
            ).add_to(map_obj)
            arrows_added += 1

        print_step("Карта", f"Добавлено {arrows_added} стрелок между {len(checkpoint_positions)} чекпоинтами")

    def _add_optimized_info_markers(self, map_obj, points, names, elevations, descriptions, colormap, smooth_route):
        """Добавляет чекпоинты, равномерно распределенные по длине сплайна"""
        if len(smooth_route) < 2:
            return

        # 1. Рассчитываем кумулятивные расстояния вдоль сплайна
        distances = [0.0]
        for i in range(1, len(smooth_route)):
            p1 = GeoPoint(*smooth_route[i-1])
            p2 = GeoPoint(*smooth_route[i])
            distances.append(distances[-1] + p1.distance_to(p2))
        
        total_length = distances[-1]

        # 2. Определяем количество и позиции чекпоинтов
        min_distance = 50  # Минимальное расстояние между чекпоинтами (метров)
        target_markers = min(20, max(10, int(total_length / 200)))  # 10-20 чекпоинтов (~каждые 200м)
        
        # 3. Выбираем точки на сплайне с равными интервалами
        spline_indices = []
        step_length = total_length / target_markers
        
        for i in range(target_markers + 1):
            target_dist = i * step_length
            closest_idx = min(range(len(distances)), key=lambda x: abs(distances[x] - target_dist))
            if not spline_indices or closest_idx != spline_indices[-1]:
                spline_indices.append(closest_idx)

        # 4. Фильтруем точки, слишком близкие друг к другу
        filtered_indices = [spline_indices[0]]
        for idx in spline_indices[1:]:
            if distances[idx] - distances[filtered_indices[-1]] >= min_distance:
                filtered_indices.append(idx)

        # 5. Находим ближайшие исходные точки для данных о высотах
        original_indices = []
        for spline_idx in filtered_indices:
            closest_original = min(
                range(len(points)),
                key=lambda x: GeoPoint(*points[x]).distance_to(GeoPoint(*smooth_route[spline_idx]))
            )
            original_indices.append(closest_original)

        # 6. Анализируем перепады высот между чекпоинтами
        segment_info = []
        for i in range(1, len(original_indices)):
            start_idx = original_indices[i-1]
            end_idx = original_indices[i]
            
            # Анализируем участок между чекпоинтами
            segment_points = points[start_idx:end_idx+1]
            segment_elev = elevations[start_idx:end_idx+1]
            segment_length = distances[filtered_indices[i]] - distances[filtered_indices[i-1]]
            
            max_elev = max(segment_elev)
            min_elev = min(segment_elev)
            start_elev = elevations[start_idx]
            end_elev = elevations[end_idx]

            # Определяем тип рельефа
            if (max_elev - min_elev) > 3 and segment_length < 100:  # Овраг
                terrain_type = "Овраг"
            elif (max_elev - min_elev) > 10:  # Значительный перепад
                terrain_type = "Холмистый участок"
            else:
                terrain_type = "Ровная местность"

            segment_info.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'max_positive': max_elev - start_elev,
                'max_negative': min_elev - start_elev,
                'net_change': end_elev - start_elev,
                'distance': segment_length,
                'terrain': terrain_type
            })
            
        checkpoints = []

        # 7. Добавляем маркеры на карту
        for i, idx in enumerate(original_indices):
            point = GeoPoint(points[idx][0], points[idx][1])
            checkpoints.append(point)
            name = names[idx] if idx < len(names) else f"Точка {i+1}"
            elevation = elevations[idx]
            desc = descriptions[idx] if idx < len(descriptions) else ""

            # Формируем описание
            if i == 0:
                marker_type = "start"
                delta_info = "Начало маршрута"
                terrain_desc = ""
            elif i == len(original_indices)-1:
                marker_type = "end"
                delta_info = f"Финиш ({segment_info[-1]['distance']:.0f} м)"
                terrain_desc = segment_info[-1]['terrain']
            else:
                marker_type = "checkpoint"
                seg = segment_info[i-1]
                delta_info = self._format_elevation_changes(seg)
                terrain_desc = seg['terrain']
                desc = f"{terrain_desc}. {desc}" if desc else terrain_desc

            # Создаем popup
            popup_content = self._create_marker_popup(
                name, i+1, len(original_indices),
                get_photo_html(point.lat, point.lon),
                desc, colormap(elevation), elevation,
                delta_info
            )

            # Добавляем маркер
            folium.Marker(
                location=points[idx],
                popup=folium.Popup(popup_content, max_width=300),
                icon=self._get_marker_icon(marker_type)
            ).add_to(map_obj)

        print_step("Карта", 
            f"Добавлено {len(original_indices)} чекпоинтов (длина маршрута {total_length:.0f} м)")
        
        return checkpoints

    def _format_elevation_changes(self, segment):
        """Форматирует информацию о перепадах высот"""
        if segment['max_positive'] > 0 and segment['max_negative'] < 0:
            return (f"↑+{segment['max_positive']:.1f}м ↓{segment['max_negative']:.1f}м "
                f"(общий: {segment['net_change']:.1f}м за {segment['distance']:.0f}м)")
        elif segment['max_positive'] > 0:
            return f"↑+{segment['max_positive']:.1f}м (общий: {segment['net_change']:.1f}м)"
        elif segment['max_negative'] < 0:
            return f"↓{segment['max_negative']:.1f}м (общий: {segment['net_change']:.1f}м)"
        return f"→ ровно ({segment['net_change']:.1f}м)"

    def _create_marker_popup(self, name, point_num, total_points, photo_html, desc, color, elevation, delta_info):
        """Создает HTML-контент для popup"""
        return f"""
        <div style="width:260px;font-family:Arial,sans-serif">
            <h4 style="margin:0;color:#333;border-bottom:1px solid #eee;padding-bottom:5px">
                {name} (точка {point_num}/{total_points})
            </h4>
            {photo_html}
            <p style="color:#666;font-size:0.9em;margin:5px 0">{desc}</p>
            <div style="background:{color};height:4px;margin:5px 0"></div>
            <p style="margin:3px 0"><b>Высота:</b> {elevation:.1f} м</p>
            <p style="margin:3px 0">
                <b>Перепады:</b> <span style="color:{"#ff0000" if '↑' in delta_info else "#00aa00" if '↓' in delta_info else "#666"}">
                {delta_info}</span>
            </p>
        </div>
        """

    def _get_marker_icon(self, marker_type):
        """Возвращает иконку для маркера"""
        icons = {
            "start": folium.Icon(color='red', icon='flag', prefix='fa'),
            "end": folium.Icon(color='darkgreen', icon='flag', prefix='fa'),
            "checkpoint": folium.Icon(color='blue', icon='info-sign', prefix='fa')
        }
        return icons.get(marker_type, folium.Icon(color='blue', icon='info-sign'))

if __name__ == '__main__':
    # Создание карты с маршрутами
    app = QApplication(sys.argv)
    window = BitsevskyMapWindow()
    window.show()
    sys.exit(app.exec_())