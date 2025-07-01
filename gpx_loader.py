import gpxpy
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from route import *
from map_helpers import print_step, safe_get_elevation

# Настройки
GPX_DIR = Path("local_routes")  # Папка с маршрутами

class LocalGPXLoader:
    def __init__(self):
        print_step("GPX", "Инициализация загрузчика локальных маршрутов")
        GPX_DIR.mkdir(exist_ok=True)

    def _extract_metadata(self, gpx) -> Dict:
        """Извлекает метаданные из GPX файла"""
        metadata = {
            "name": "Безымянный маршрут",
            "desc": "",
            "time": None
        }
        
        if gpx.name:
            metadata["name"] = gpx.name
        elif gpx.tracks and gpx.tracks[0].name:
            metadata["name"] = gpx.tracks[0].name
            
        if gpx.description:
            metadata["desc"] = gpx.description
        elif gpx.tracks and gpx.tracks[0].description:
            metadata["desc"] = gpx.tracks[0].description
            
        if gpx.time:
            if gpx.time.tzinfo is None:
                metadata["time"] = gpx.time.replace(tzinfo=timezone.utc)
            else:
                metadata["time"] = gpx.time
        return metadata

    def load_routes(self) -> List[Route]:
        """Загружает все валидные маршруты из локальной папки"""
        print_step("GPX", f"Поиск GPX-файлов в {GPX_DIR}")
        routes = []
        for gpx_file in GPX_DIR.glob("*.gpx"):
            try:
                with open(gpx_file, 'r', encoding='utf-8') as f:
                    gpx = gpxpy.parse(f)
                
                metadata = self._extract_metadata(gpx)
                points = []
                elevations = []
                descriptions = []

                start_time_route = None
                end_time_route = None
                
                for track in gpx.tracks:
                    for segment in track.segments:
                        for i, point in enumerate(segment.points):
                            point_time = point.time
                            if point_time:
                                # Convert to UTC if it has timezone, otherwise assume UTC
                                if point_time.tzinfo is None:
                                    point_time = point_time.replace(tzinfo=timezone.utc)
                                else:
                                    point_time = point_time.astimezone(timezone.utc)

                            if i == 0:
                                if not start_time_route:
                                    start_time_route = point_time
                                elapsed_seconds = 0.0
                            else:
                                if start_time_route and point_time:
                                    elapsed_seconds = (point_time - start_time_route).total_seconds()
                                else:
                                    print_step("GPX", f"Внимание: Отсутствует время для точки {point.latitude}, {point.longitude}. Точка будет пропущена.", level="WARNING")
                                    continue

                            gp = GeoPoint(
                                lat=point.latitude,
                                lon=point.longitude,
                                time=point_time,
                                elapsed_seconds=elapsed_seconds
                            )
                            points.append(gp)
                            elevations.append(point.elevation if point.elevation else safe_get_elevation(point.latitude, point.longitude))
                            descriptions.append(point.description if point.description else "")
                            end_time_route = point_time

                if not points:
                    print_step("GPX", f"В файле {gpx_file.name} не найдено валидных точек и будет пропущен.")
                    continue

                # Ensure start and end times are in UTC
                if start_time_route and start_time_route.tzinfo is None:
                    start_time_route = start_time_route.replace(tzinfo=timezone.utc)
                if end_time_route and end_time_route.tzinfo is None:
                    end_time_route = end_time_route.replace(tzinfo=timezone.utc)

                route = Route(
                    name=metadata['name'],
                    points=points,
                    elevations=elevations,
                    descriptions=descriptions,
                    start_time=start_time_route,
                    end_time=end_time_route
                )
                routes.append(route)
                        
            except Exception as e:
                print_step("GPX", f"Ошибка обработки {gpx_file.name}: {e}")
        
        return routes

if __name__ == "__main__":
    loader = LocalGPXLoader()
    routes = loader.load_routes()
    
    if routes:
        print("\nЗагруженные маршруты:")
        for i, route in enumerate(routes, 1):
            print(f"{i}. {route.name}")
            print(f"   Точек: {len(route.points)}")
            if route.points:
                print(f"   Первая точка: ({route.points[0].lat}, {route.points[0].lon})")
                print(f"   Время старта: {route.start_time}")
                print(f"   Последняя точка: ({route.points[-1].lat}, {route.points[-1].lon})")
                print(f"   Время финиша: {route.end_time}")
                print(f"   Длительность: {timedelta(seconds=route.points[-1].elapsed_seconds)}")
            else:
                print("   Точек нет.")
    else:
        print("Не найдено ни одного подходящего маршрута")
        print(f"Поместите GPX-файлы в папку {GPX_DIR}")