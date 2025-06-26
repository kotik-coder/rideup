import gpxpy
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import math
from datetime import datetime

# Настройки
GPX_DIR = Path("local_routes")  # Папка с маршрутами
BITZA_BOUNDS = {
    'min_lat': 55.57,
    'max_lat': 55.61,
    'min_lon': 37.52,
    'max_lon': 37.58
}
MAX_DISTANCE_KM = 3.0  # Максимальное расстояние от границ леса

@dataclass
class BikeRoute:
    name: str
    points: List[Tuple[float, float]]  # (lat, lon)
    elevations: List[float]
    descriptions: List[str]
    source: str = "local"

    def to_map_format(self) -> Dict:
        return {
            "points": self.points,
            "names": [self.name] * len(self.points),
            "elevations": self.elevations,
            "descriptions": self.descriptions,
            "metadata": {
                "name": self.name,
                "source": self.source,
                "url": ""
            }
        }

class LocalGPXLoader:
    def __init__(self):
        self._print_step("Инициализация загрузчика локальных маршрутов")
        GPX_DIR.mkdir(exist_ok=True)

    def _print_step(self, message: str):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] [GPX] {message}")

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
            metadata["time"] = gpx.time.isoformat()
            
        return metadata

    def _point_in_bounds(self, point: Tuple[float, float]) -> bool:
        """Проверяет, находится ли точка в границах Битцевского леса или не далее MAX_DISTANCE_KM от них"""
        lat, lon = point
        
        # Проверка нахождения внутри прямоугольника
        if (BITZA_BOUNDS['min_lat'] <= lat <= BITZA_BOUNDS['max_lat'] and
            BITZA_BOUNDS['min_lon'] <= lon <= BITZA_BOUNDS['max_lon']):
            return True
        
        # Если точка вне прямоугольника, проверяем расстояние до границ
        # Вычисляем расстояния до каждой границы в километрах
        dist_north = max(0, lat - BITZA_BOUNDS['max_lat']) * 111.32
        dist_south = max(0, BITZA_BOUNDS['min_lat'] - lat) * 111.32
        dist_east = max(0, lon - BITZA_BOUNDS['max_lon']) * 111.32 * math.cos(math.radians(lat))
        dist_west = max(0, BITZA_BOUNDS['min_lon'] - lon) * 111.32 * math.cos(math.radians(lat))
        
        # Находим минимальное расстояние до границы
        min_distance = min(dist_north, dist_south, dist_east, dist_west)
        
        return min_distance <= MAX_DISTANCE_KM
        
    def _is_valid_route(self, points: List[Tuple[float, float]]) -> bool:
        """Проверяет что маршрут полностью находится в допустимой зоне"""
        if not points:
            return False
            
        return all(self._point_in_bounds(point) for point in points)

    def load_routes(self) -> List[Dict]:
        """Загружает все валидные маршруты из локальной папки"""
        self._print_step(f"Поиск GPX-файлов в {GPX_DIR}")
        valid_routes = []
        
        for gpx_file in GPX_DIR.glob("*.gpx"):
            try:
                self._print_step(f"Обработка файла: {gpx_file.name}")
                with open(gpx_file, 'r', encoding='utf-8') as f:
                    gpx = gpxpy.parse(f)
                    metadata = self._extract_metadata(gpx)
                    
                    points = []
                    elevations = []
                    descriptions = []
                    
                    for track in gpx.tracks:
                        for segment in track.segments:
                            for point in segment.points:
                                points.append((point.latitude, point.longitude))
                                elevations.append(point.elevation or 0)
                                desc = f"{metadata['name']} - {point.time}" if point.time else metadata['name']
                                descriptions.append(desc)
                    
                    if self._is_valid_route(points):
                        route = BikeRoute(
                            name=metadata['name'],
                            points=points,
                            elevations=elevations,
                            descriptions=descriptions
                        )
                        valid_routes.append(route.to_map_format())
                        self._print_step(f"Маршрут {metadata['name']} добавлен")
                    else:
                        self._print_step(f"Маршрут {metadata['name']} вне зоны Битцевского леса")
                        
            except Exception as e:
                self._print_step(f"Ошибка обработки {gpx_file.name}: {e}")
        
        self._print_step(f"Найдено {len(valid_routes)} валидных маршрутов")
        return valid_routes

if __name__ == "__main__":
    loader = LocalGPXLoader()
    routes = loader.load_routes()
    
    if routes:
        print("\nЗагруженные маршруты:")
        for i, route in enumerate(routes, 1):
            print(f"{i}. {route['metadata']['name']}")
            print(f"   Точек: {len(route['points'])}")
            print(f"   Первая точка: {route['points'][0]}")
            print(f"   Последняя точка: {route['points'][-1]}")
    else:
        print("Не найдено ни одного подходящего маршрута")
        print(f"Поместите GPX-файлы в папку {GPX_DIR}")