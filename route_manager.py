import osmnx as ox
import numpy as np
from scipy.interpolate import Akima1DInterpolator, interp1d
from datetime import datetime, timezone

from map_helpers import point_to_segment_projection_and_distance, print_step, safe_get_elevation, get_boundary_point, generate_nearby_point, get_landscape_description, get_boundary_near_point
from gpx_loader import *
import gpx_loader
from route import GeoPoint
from media_helpers import *
from typing import List, Tuple, Dict, Optional # Added Optional for clarity
from dataclasses import dataclass
from shapely.geometry import Point, Polygon
from pathlib import Path

from track import Track, TrackPoint

PHOTO_CHECKPOINT_DISTANCE_THRESHOLD = 50.0 # meters

class RouteManager():

    bounds : List[float]
    routes : List[Route]
    tracks : List[Track]
    local_photos : List[Dict[str, any]]
    local_photos_folder: str
    _photos_loaded: bool

    def __init__(self, geostring : str, local_photos_folder: str = "local_photos"):
        self.bounds = self.get_forest_bounds(geostring)
        self.local_photos_folder = local_photos_folder 
        self.local_photos = [] 
        self._photos_loaded = False 
        self.routes: List[Route] = []
        self.tracks: List[Track] = [] # Initialize as list of Track objects
        self._route_to_tracks: Dict[str, List[Track]] = {} # Initialize the association dictionary
        print_step("Route Manager", "Инициализация RouteManager завершена.")

    def _load_local_photos(self, folder_path_str: str) -> List[Dict[str, any]]:
        """Loads local photos with geolocation and timestamp data from a specified folder."""
        if self._photos_loaded:
            return self.local_photos

        print_step("Маршруты", f"Загружаю локальные фото из папки: {folder_path_str}...")
        folder_path = Path(folder_path_str)
        if not folder_path.is_dir():
            print_step("Маршруты", f"Папка '{folder_path_str}' не найдена или не является директорией.", level="WARNING")
            return []
        
        photos = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for image_path in folder_path.glob(ext):
                try:
                    timestamp = get_photo_timestamp(image_path)                    
                    if timestamp:
                        photos.append({
                            'path': str(image_path),
                            'coords': get_exif_geolocation(image_path),
                            'timestamp': timestamp
                        })
                except Exception:
                    continue
        
        print_step("Маршруты", f"Найдено {len(photos)} фото с временными метками.")
        self.local_photos = photos
        self._photos_loaded = True
        return self.local_photos

    def _create_velocity_profile(self, track: Track) -> List[Dict[str, float]]:
        """
        Creates a velocity profile from a Track object's analysis data.
        Returns a list of dictionaries, each with 'distance' and 'velocity'.
        """
        profile = []
        if not track.analysis:
            print_step("Процессинг", f"Создание профиля скорости: Трек не содержит данных анализа. Возвращаю пустой профиль.")
            return []

        for analysis_point in track.analysis:
            profile.append({
                'distance': analysis_point.distance_from_start,
                'velocity': analysis_point.speed
            })
        print_step("Процессинг", f"Профиль скорости для трека создан успешно.")
        return profile

    def load_valid_routes(self):
        print_step("Маршруты", "Загружаю маршруты и треки...")
        loader = LocalGPXLoader()
        
        # Collect all (Route, List[Track]) pairs from all GPX files
        loaded_route_track_pairs: List[Tuple[Route, List[Track]]] = []
        # Ensure gpx_loader is properly imported and GPX_DIR is accessible
        import gpx_loader 
        for gpx_file in gpx_loader.GPX_DIR.glob("*.gpx"):
            try:
                # loader.load_routes_and_tracks now returns List[Tuple[Route, List[Track]]]
                # Use .extend() as load_routes_and_tracks returns a list of tuples for each file
                loaded_route_track_pairs.extend(loader.load_routes_and_tracks(gpx_file))
            except Exception as e:
                print_step("Маршруты", f"Ошибка при загрузке GPX файла {gpx_file.name}: {e}", level="ERROR")
                continue

        self.routes = [] # This will store only Route objects
        self.tracks = [] # This will be a flattened list of all valid Track objects
        self._route_to_tracks = {} # Maps route names to their associated tracks

        for route, associated_tracks in loaded_route_track_pairs:
            # Filter routes based on validity criteria (e.g., within bounds)
            if route.is_valid_route(self.bounds): # Assuming is_valid_route is a method of Route
                self.routes.append(route) # Add the Route object to the list of routes
                # Add all tracks associated with this valid route to the flat list of all tracks
                self.tracks.extend(associated_tracks) 
                # Store the association: route name to its list of associated tracks
                self._route_to_tracks[route.name] = associated_tracks 
            else:
                print_step("Маршруты", f"Маршрут '{route.name}' вне границ леса или невалиден. Пропускаю.", level="WARN")

        print_step("Маршруты", f"Найдено {len(self.routes)} валидных маршрутов")
        print_step("Маршруты", f"Найдено {len(self.tracks)} треков")
        return self.routes

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

    def _process_route(self, route: Route):
        print_step("Процессинг", f"Начинаю обработку маршрута: {route.name}")

        smooth_points, smooth_elevations = self._create_smooth_route(route)
        print_step("Процессинг", f"Маршрут '{route.name}': Получено {len(smooth_points)} сглаженных точек.")

        loaded_photos = self._load_local_photos(self.local_photos_folder)

        associated_tracks = self._route_to_tracks.get(route.name, [])

        # Calculate distances for checkpoints first
        distances = [0.0]
        for i in range(1, len(smooth_points)):
            p1 = GeoPoint(*smooth_points[i-1])
            p2 = GeoPoint(*smooth_points[i])
            distances.append(distances[-1] + p1.distance_to(p2))

        # Generate checkpoints (without full descriptions initially)
        checkpoints = self._get_checkpoints(route, smooth_points, smooth_elevations, loaded_photos, associated_tracks)
        print_step("Процессинг", f"Маршрут '{route.name}': Получено {len(checkpoints)} чекпоинтов.")

        # Assign distances to checkpoints
        if smooth_points and checkpoints:
            total_dist_so_far = 0
            current_smooth_idx = 0
            for cp in checkpoints:
                target_smooth_idx = cp['point_index']
                for k in range(current_smooth_idx, min(target_smooth_idx, len(smooth_points) - 1)):
                    p1 = GeoPoint(*smooth_points[k])
                    p2 = GeoPoint(*smooth_points[k+1])
                    total_dist_so_far += p1.distance_to(p2)
                cp['distance_from_start'] = total_dist_so_far
                current_smooth_idx = target_smooth_idx
        else:
            print_step("Процессинг", f"Маршрут '{route.name}': Пропущен расчет расстояний чекпоинтов из-за отсутствия сглаженных точек или чекпоинтов.")

        # Now, calculate segments and then enrich checkpoint descriptions
        segments = self._calculate_segments(checkpoints, smooth_elevations)
        print_step("Процессинг", f"Маршрут '{route.name}': Получено {len(segments)} сегментов.")

        # Enrich checkpoint descriptions using segment information
        # Iterate through checkpoints (excluding the very last one, as it's an end point of a segment)
        for i in range(len(checkpoints)):
            cp = checkpoints[i]
            if i < len(segments): # For start and intermediate points of segments
                segment_info = segments[i]
                # Pass current checkpoint elevation and segment stats to get_landscape_description
                cp['description'] = get_landscape_description(
                    current_elevation=cp['elevation'],
                    segment_net_elevation_change=segment_info['net_elevation'],
                    segment_elevation_gain=segment_info['elevation_gain'],
                    segment_elevation_loss=segment_info['elevation_loss'],
                    segment_distance=segment_info['distance']
                )
            elif i == len(checkpoints) - 1 and len(checkpoints) > 1:
                # This is the very last checkpoint (the finish point)
                # Its description is typically "Конец маршрута."
                cp['description'] = "Конец маршрута."
            else: # For the very first checkpoint, set a general start description
                cp['description'] = "Начало маршрута."


        elevation_profile = self._create_elevation_profile(smooth_points, smooth_elevations)
        print_step("Процессинг", f"Маршрут '{route.name}': Получен профиль высот с {len(elevation_profile)} точками.")

        velocity_profile = []
        if associated_tracks:
            velocity_profile = self._create_velocity_profile(associated_tracks[0])
            print_step("Процессинг", f"Маршрут '{route.name}': Получен профиль скорости с {len(velocity_profile)} точками.")

        processed_route_dict = {
            'name': route.name,
            'checkpoints': checkpoints,
            'segments': segments,
            'elevation_profile': elevation_profile,
            'velocity_profile': velocity_profile,
            'raw_points': [(p.lat, p.lon) for p in route.points],
            'raw_elevations': route.elevations,
            'smooth_points': smooth_points
        }

        print_step("Процессинг", f"Заканчиваю обработку маршрута: {route.name}.")
        return processed_route_dict

    def _create_elevation_profile(self, points: List[Tuple[float, float]], elevations: List[float]):
        profile = []
        total_distance = 0
        if not points or not elevations:
            print_step("Процессинг", "Создание профиля высот: Пустые точки или высоты. Возвращаю пустой профиль.")
            return []

        profile.append({'distance': 0, 'elevation': elevations[0]})
        for i in range(1, len(points)):
            p1 = GeoPoint(points[i-1][0], points[i-1][1])
            p2 = GeoPoint(points[i][0], points[i][1])
            total_distance += p1.distance_to(p2)
            profile.append({'distance': total_distance, 'elevation': elevations[i]})
        return profile

    def _get_checkpoints(self, route: Route, smooth_route: List[Tuple[float, float]], 
                        route_elevations: List[float], local_photos: List[Dict[str, any]],
                        associated_tracks: List[Track]):
        if len(smooth_route) < 2:
            print_step("Процессинг", "Получение чекпоинтов: Слишком мало сглаженных точек. Возвращаю пустой список.")
            return []

        # Calculate cumulative distances along the smooth route
        distances = [0.0]
        for i in range(1, len(smooth_route)):
            p1 = GeoPoint(*smooth_route[i-1])
            p2 = GeoPoint(*smooth_route[i])
            distances.append(distances[-1] + p1.distance_to(p2))

        total_length = distances[-1]

        marker_indices = set()
        index_to_photo_info: Dict[int, Dict[str, any]] = {} 

        marker_indices.add(0)
        marker_indices.add(len(smooth_route) - 1)

        for track in associated_tracks:
            if not track.points:
                continue

            track_start_time = track.points[0].timestamp if track.points else None
            track_end_time = track.points[-1].timestamp if track.points else None

            if track_start_time and track_end_time:
                track_duration = (track_end_time - track_start_time).total_seconds()
                
                local_photos.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min.replace(tzinfo=timezone.utc))

                for photo in local_photos:
                    if photo['path'] in [info.get('path') for info in index_to_photo_info.values()]:
                        continue 

                    photo_time = photo.get('timestamp')
                    if photo_time:
                        photo_elapsed = (photo_time - track_start_time).total_seconds()
                        
                        if 0 <= photo_elapsed <= track_duration:
                            closest_track_point: Optional[TrackPoint] = None
                            min_time_diff = float('inf')
                            
                            for tp in track.points:
                                time_diff = abs(photo_elapsed - tp.elapsed_seconds)
                                if time_diff < min_time_diff:
                                    min_time_diff = time_diff
                                    closest_track_point = tp
                            
                            if closest_track_point is not None and min_time_diff < 300:
                                closest_smooth_idx = min(
                                    range(len(smooth_route)),
                                    key=lambda x: GeoPoint(*smooth_route[x]).distance_to(closest_track_point.point)
                                )
                                marker_indices.add(closest_smooth_idx)
                                index_to_photo_info[closest_smooth_idx] = photo


        target_markers_uniform = min(20, max(5, int(total_length / 250))) if total_length > 0 else 2
        if target_markers_uniform > 2 and total_length > 0:
            step_length = total_length / (target_markers_uniform - 1)
            for i in range(1, target_markers_uniform - 1):
                target_dist = i * step_length
                closest_idx = min(range(len(distances)), key=lambda x: abs(distances[x] - target_dist))
                marker_indices.add(closest_idx)

        sorted_indices = sorted(list(marker_indices))
        
        checkpoints = []

        for i, idx in enumerate(sorted_indices):
            point = smooth_route[idx]
            elevation = route_elevations[idx]
            
            photo_info = index_to_photo_info.get(idx)

            point_name = f"Точка {i+1}"
            # description is set AFTER segments are calculated in _process_route
            description = "" 
            photo_html = ""

            if photo_info:
                point_name = f"Фототочка"
                photo_html = get_photo_html(point[0], point[1], local_photo_path=photo_info['path'])
            else:
                photo_html = get_photo_html(point[0], point[1])

            if i == 0:
                point_name = "Старт"
                description = "Начало маршрута." # Initial description for start
            elif i == len(sorted_indices) - 1:
                point_name = "Финиш"
                description = "Конец маршрута." # Initial description for finish


            checkpoint = {
                'point_index': idx,
                'position': i + 1,
                'total_positions': len(sorted_indices),
                'lat': point[0],
                'lon': point[1],
                'elevation': elevation,
                'name': point_name,
                'description': description, # Will be overwritten later for intermediate points
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

            # Calculate min and max elevation within the segment
            min_seg_elevation = min(segment_elevations) if segment_elevations else 0.0
            max_seg_elevation = max(segment_elevations) if segment_elevations else 0.0
            
            segments.append({
                'distance': end_cp.get('distance_from_start', 0) - start_cp.get('distance_from_start', 0),
                'elevation_gain': max(0, max_seg_elevation - start_cp['elevation']), # This is the net gain from start_cp
                'elevation_loss': max(0, start_cp['elevation'] - min_seg_elevation), # This is the net loss from start_cp
                'net_elevation': end_cp['elevation'] - start_cp['elevation'],
                'min_segment_elevation': min_seg_elevation, # Add min elevation for the segment
                'max_segment_elevation': max_seg_elevation, # Add max elevation for the segment
                'start_checkpoint': start_cp['position'] - 1,
                'end_checkpoint': end_cp['position'] - 1
            })
        return segments


    def _create_smooth_route(self, route: Route):
        # Thresholds for interpolation method selection
        MIN_POINTS_FOR_LINEAR = 50     # Use linear if ≥50 points
        MAX_DISTANCE_FOR_LINEAR = 10.0 # Use linear if avg spacing <10m
        
        if len(route.points) < 4:
            print_step("Smoothing", f"Route '{route.name}': <4 points, returning raw points")
            return [(p.lat, p.lon) for p in route.points], route.elevations

        points = np.array([(p.lat, p.lon) for p in route.points])
        elevations = np.array(route.elevations)

        # Calculate cumulative distances and average spacing
        distances = np.zeros(len(points))
        for i in range(1, len(points)):
            p1 = GeoPoint(points[i-1][0], points[i-1][1])
            p2 = GeoPoint(points[i][0], points[i][1])
            distances[i] = distances[i-1] + p1.distance_to(p2)
        
        avg_distance = distances[-1] / (len(points) - 1) if len(points) > 1 else 0
        
        # Determine interpolation method
        use_linear = (len(points) >= MIN_POINTS_FOR_LINEAR and 
                    avg_distance <= MAX_DISTANCE_FOR_LINEAR)
        
        method = "linear" if use_linear else "Akima"
        print_step("Smoothing", 
                f"Route '{route.name}': {len(points)} points, "
                f"avg spacing {avg_distance:.1f}m → {method} interpolation")

        if distances[-1] == 0:
            return [(p.lat, p.lon) for p in route.points], route.elevations

        # Normalized distance parameter (0-1)
        t = distances / distances[-1]
        num_smooth_points = max(100, int(distances[-1] / 10))
        t_smooth = np.linspace(0, 1, num_smooth_points)

        try:
            if use_linear:
                # Linear interpolation - preserves sharp turns in dense data
                interp_lat  = interp1d(t, points[:, 0], kind='linear')
                interp_lon  = interp1d(t, points[:, 1], kind='linear')
                interp_elev = interp1d(t, elevations, kind='linear')
            else:
                # Akima spline - smooths sparse data
                interp_lat = Akima1DInterpolator(t, points[:, 0])
                interp_lon = Akima1DInterpolator(t, points[:, 1])
                interp_elev = Akima1DInterpolator(t, elevations)

            smooth_points = list(zip(
                interp_lat(t_smooth),
                interp_lon(t_smooth)
            ))
            return smooth_points, interp_elev(t_smooth).tolist()

        except Exception as e:
            print_step("Error", f"Smoothing failed for {route.name}: {str(e)}")
            return [(p.lat, p.lon) for p in route.points], route.elevations