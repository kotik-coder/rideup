import osmnx as ox
import numpy as np
from scipy.interpolate import Akima1DInterpolator

from datetime import datetime

from map_helpers import print_step, safe_get_elevation, get_boundary_point, generate_nearby_point, get_landscape_description, get_boundary_near_point
from gpx_loader import LocalGPXLoader, Route
from route import GeoPoint
from media_helpers import *
from typing import List, Tuple, Dict
from dataclasses import dataclass
from shapely.geometry import Point, Polygon
from pathlib import Path

class RouteManager():

    bounds       : List[float]
    valid_routes : List[Route]
    local_photos : List[Dict[str, any]]
    local_photos_folder: str # <--- ADDED LINE: To store the folder path
    _photos_loaded: bool # <--- ADDED LINE: Flag to ensure photos are loaded only once

    def __init__(self, geostring : str, local_photos_folder: str = "local_photos"):
        self.bounds = self.get_forest_bounds(geostring)
        self.local_photos_folder = local_photos_folder # <--- MODIFIED LINE: Store the folder path
        self.local_photos = [] # <--- MODIFIED LINE: Initialize as empty
        self._photos_loaded = False # <--- ADDED LINE: Initialize flag
        self.valid_routes = self.load_valid_routes()

    def _load_local_photos(self, folder_path_str: str) -> List[Dict[str, any]]:
        """Loads local photos with geolocation data from a specified folder."""
        # This method is now called only when needed, and only once.
        if self._photos_loaded: # <--- ADDED CHECK
            return self.local_photos

        print_step("Маршруты", f"Загружаю локальные фото из папки: {folder_path_str}...")
        folder_path = Path(folder_path_str)
        if not folder_path.is_dir():
            print_step("Маршруты", f"Папка '{folder_path_str}' не найдена или не является директорией.", level="WARNING")
            return []
        photos = get_photos_with_geolocation_from_folder(folder_path)
        print_step("Маршруты", f"Найдено {len(photos)} фото с геометками.")
        self.local_photos = photos # Store for future use within this instance
        self._photos_loaded = True # Set flag to true
        return self.local_photos

    def load_valid_routes(self):
        print_step("Маршруты", "Загружаю маршруты...")
        loader = LocalGPXLoader()
        valid_routes = [ r \
                            for r in loader.load_routes() \
                            if r.is_valid_route(self.bounds) ]
        print_step("Маршруты", f"Найдено {len(valid_routes)} валидных маршрутов")
        return valid_routes

    def get_forest_bounds(self, geostring: str) -> List[float]:
        """
        Получает границы леса с помощью OSMnx.
        Возвращает список [min_lon, min_lat, max_lon, max_lat].
        """
        print_step(prefix="Маршруты", message=f"Получаю границы области {geostring}...")
        try:
            forest = ox.geocode_to_gdf(geostring)
            bounds = [float(x) for x in forest.total_bounds]
            print_step("Маршруты", f"Границы получены: {bounds}")
            return bounds
        except Exception as e:
            print_step("Ошибка", f"Не удалось получить границы для {geostring}: {e}")
            return [37.5, 55.5, 37.6, 55.6] # Example default bounds if OSMnx fails

    def generate_route(self, polygon: Polygon, bounds: List[float]) -> Tuple[List[Tuple[float, float]], List[str], List[float], List[str]]:
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
            candidate = get_boundary_near_point(polygon, bounds, points[-1], 500)
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

    # --- Route Data Processing Methods (Moved from map.py and integrated) ---
    def _process_route(self, route: Route):
        print_step("Процессинг", f"Начинаю обработку маршрута: {route.name}")

        smooth_points, smooth_elevations = self._create_smooth_route(route)
        print_step("Процессинг", f"Маршрут '{route.name}': Получено {len(smooth_points)} сглаженных точек.")

        # Ensure local photos are loaded before getting checkpoints
        # This will load photos only once, the first time _process_route is called
        loaded_photos = self._load_local_photos(self.local_photos_folder) # <--- MODIFIED LINE

        # Pass local_photos to _get_checkpoints
        checkpoints = self._get_checkpoints(route, smooth_points, smooth_elevations, loaded_photos) # <--- MODIFIED LINE
        print_step("Процессинг", f"Маршрут '{route.name}': Получено {len(checkpoints)} чекпоинтов.")

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
            print_step("Процессинг", f"Маршрут '{route.name}': Расстояния чекпоинтов рассчитаны.")
        else:
            print_step("Процессинг", f"Маршрут '{route.name}': Пропущен расчет расстояний чекпоинтов из-за отсутствия сглаженных точек или чекпоинтов.")


        segments = self._calculate_segments(checkpoints, smooth_elevations)
        print_step("Процессинг", f"Маршрут '{route.name}': Получено {len(segments)} сегментов.")

        elevation_profile = self._create_elevation_profile(smooth_points, smooth_elevations)
        print_step("Процессинг", f"Маршрут '{route.name}': Получен профиль высот с {len(elevation_profile)} точками.")

        processed_route_dict = {
            'name': route.name,
            'checkpoints': checkpoints,
            'segments': segments,
            'elevation_profile': elevation_profile,
            'raw_points': [(p.lat, p.lon) for p in route.points],
            'raw_elevations': route.elevations,
            'smooth_points': smooth_points
        }

        print_step("Процессинг", f"Заканчиваю обработку маршрута: {route.name}. Возвращаю данные (первые 300 символов): {str(processed_route_dict)[:300]}")
        return processed_route_dict

    def _create_elevation_profile(self, points: List[Tuple[float, float]], elevations: List[float]):
        profile = []
        total_distance = 0
        if not points or not elevations:
            print_step("Процессинг", "Создание профиля высот: Пустые точки или высоты. Возвращаю пустой профиль.")
            return []

        profile.append({'distance': 0, 'elevation': elevations[0]})
        for i in range(1, len(points)):
            p1 = GeoPoint(*points[i-1])
            p2 = GeoPoint(*points[i])
            total_distance += p1.distance_to(p2)
            profile.append({'distance': total_distance, 'elevation': elevations[i]})
        return profile

    def _get_checkpoints(self, route: Route, smooth_route: List[Tuple[float, float]], route_elevations: List[float], local_photos: List[Dict[str, any]]):
        if len(smooth_route) < 2:
            print_step("Процессинг", "Получение чекпоинтов: Слишком мало сглаженных точек. Возвращаю пустой список.")
            return []

        distances = [0.0]
        for i in range(1, len(smooth_route)):
            p1 = GeoPoint(*smooth_route[i-1])
            p2 = GeoPoint(*smooth_route[i])
            distances.append(distances[-1] + p1.distance_to(p2))

        total_length = distances[-1]

        marker_indices = set()
        # Always include start and end
        marker_indices.add(0)
        marker_indices.add(len(smooth_route) - 1)

        # Add uniformly spaced checkpoints
        target_markers_uniform = min(20, max(5, int(total_length / 250))) if total_length > 0 else 2
        if target_markers_uniform > 2 and total_length > 0:
            step_length = total_length / (target_markers_uniform - 1)
            for i in range(1, target_markers_uniform - 1):
                target_dist = i * step_length
                closest_idx = min(range(len(distances)), key=lambda x: abs(distances[x] - target_dist))
                marker_indices.add(closest_idx)

        # Add checkpoints from local photos if they are close to the route
        PHOTO_CHECKPOINT_DISTANCE_THRESHOLD = 50.0 # meters
        for photo in local_photos:
            photo_lat, photo_lon = photo['coords']
            photo_point = GeoPoint(photo_lat, photo_lon)

            # Find the closest point on the smooth route to the photo
            min_dist_to_route = float('inf')
            closest_smooth_idx = -1
            for i, route_point_coords in enumerate(smooth_route):
                route_point = GeoPoint(route_point_coords[0], route_point_coords[1])
                dist = photo_point.distance_to(route_point)
                if dist < min_dist_to_route:
                    min_dist_to_route = dist
                    closest_smooth_idx = i

            if closest_smooth_idx != -1 and min_dist_to_route <= PHOTO_CHECKPOINT_DISTANCE_THRESHOLD:
                # Add this index as a potential checkpoint
                marker_indices.add(closest_smooth_idx)
                print_step("Процессинг", f"Добавлен чекпоинт из фото: {photo['path']} (dist: {min_dist_to_route:.2f}m)")

        sorted_indices = sorted(list(marker_indices))

        checkpoints = []
        for i, idx in enumerate(sorted_indices):
            point = smooth_route[idx]
            elevation = route_elevations[idx]

            # Find if this checkpoint corresponds to a local photo
            photo_info = None
            for photo in local_photos:
                photo_lat, photo_lon = photo['coords']
                photo_point = GeoPoint(photo_lat, photo_lon)
                cp_point = GeoPoint(point[0], point[1])
                if photo_point.distance_to(cp_point) <= PHOTO_CHECKPOINT_DISTANCE_THRESHOLD:
                    photo_info = photo
                    break

            point_name = f"Точка {i+1}"
            description = ""
            photo_html = "" # Initialize photo_html

            if i == 0:
                point_name = "Старт"
                description = "Начало маршрута."
            elif i == len(sorted_indices) - 1:
                point_name = "Финиш"
                description = "Конец маршрута."

            if photo_info:
                # Removed the filename from the point_name
                point_name = f"Фототочка"
                # Removed the specific description text
                description = "" # Or remove this line if 'description' is not used for photo checkpoints
                display_path = f"/local_photos/{Path(photo_info['path']).name}"
                photo_html = f"""
                <div style="margin:10px 0;text-align:center">
                    <img src="{display_path}"
                         alt="{Path(photo_info['path']).name}"
                         style="max-width:100%;max-height:200px;border-radius:4px;border:1px solid #eee;">
                    <p style="font-size:0.8em;color:#999;margin-top:5px">
                        Источник: Локальное фото
                    </p>
                </div>
                """
                # The rest of your checkpoint dictionary construction follows...
                # Make sure the 'description' field in the checkpoint dictionary uses this modified 'description' variable.
            else:
                # Fallback to online photo if no local photo for this checkpoint
                photo_html = get_photo_html(point[0], point[1])

            # Attempt to get a more relevant description for non-photo checkpoints
            # only if it's not start/finish or a photo checkpoint
            if not photo_info and i != 0 and i != len(sorted_indices) - 1:
                closest_original_idx = min(range(len(route.points)), key=lambda x: GeoPoint(route.points[x].lat, route.points[x].lon).distance_to(GeoPoint(*point)))
                description = route.descriptions[closest_original_idx] if closest_original_idx < len(route.descriptions) else ""


            checkpoint = {
                'point_index': idx,
                'position': i + 1,
                'total_positions': len(sorted_indices),
                'lat': point[0],
                'lon': point[1],
                'elevation': elevation,
                'name': point_name,
                'description': description,
                'photo_html': photo_html
            }
            checkpoints.append(checkpoint)

        return checkpoints

    def _calculate_segments(self, checkpoints: List[dict], elevations: List[float]):
        segments = []
        if not checkpoints or len(checkpoints) < 2:
            print_step("Процессинг", "Расчет сегментов: Нет чекпоинтов или слишком мало. Возвращаю пустой список.")
            return []

        for i in range(1, len(checkpoints)):
            start_cp = checkpoints[i-1]
            end_cp = checkpoints[i]

            start_idx = start_cp['point_index']
            end_idx = end_cp['point_index']

            segment_elevations = elevations[start_idx:end_idx+1]
            if not segment_elevations:
                print_step("Процессинг", f"Расчет сегментов: Сегмент от {start_idx} до {end_idx} не имеет данных о высоте. Пропускаю.")
                continue

            segments.append({
                'distance': end_cp.get('distance_from_start', 0) - start_cp.get('distance_from_start', 0),
                'elevation_gain': max(0, max(segment_elevations) - start_cp['elevation']),
                'elevation_loss': max(0, start_cp['elevation'] - min(segment_elevations)),
                'net_elevation': end_cp['elevation'] - start_cp['elevation'],
                'start_checkpoint': start_cp['position'] - 1,
                'end_checkpoint': end_cp['position'] - 1
            })
        return segments

    def _create_smooth_route(self, route: Route):
        if len(route.points) < 4:
            print_step("Процессинг", f"Сглаживание маршрута '{route.name}': Менее 4 точек. Возвращаю необработанные точки.")
            return [(p.lat, p.lon) for p in route.points], route.elevations

        points = np.array([(p.lat, p.lon) for p in route.points])
        elevations = np.array(route.elevations)

        distances = np.zeros(len(points))
        for i in range(1, len(points)):
            p1 = GeoPoint(points[i-1][0], points[i-1][1])
            p2 = GeoPoint(points[i][0], points[i][1])
            distances[i] = distances[i-1] + p1.distance_to(p2)

        if distances[-1] == 0:
            print_step("Процессинг", f"Сглаживание маршрута '{route.name}': Нулевая длина маршрута. Возвращаю необработанные точки.")
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
            print_step("Ошибка", f"Не удалось сгладить маршрут {route.name}, возвращаю необработанные точки. Ошибка: {e}")
            return [(p.lat, p.lon) for p in route.points], route.elevations