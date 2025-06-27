import gpxpy
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import math
from datetime import datetime
from route import *
from map_helpers import print_step

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
            metadata["time"] = gpx.time.isoformat()
            
        return metadata

    def load_routes(self) -> List[Route]:
        """Загружает все валидные маршруты из локальной папки"""
        print_step("GPX", f"Поиск GPX-файлов в {GPX_DIR}")
        routes = []
        
        for gpx_file in GPX_DIR.glob("*.gpx"):
            try:
                print_step("GPX", f"Обработка файла: {gpx_file.name}")
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
                    
                    
                    route = Route(
                        name=metadata['name'],
                        points=points,
                        elevations=elevations,
                        descriptions=descriptions
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
            print(f"{i}. {route['metadata']['name']}")
            print(f"   Точек: {len(route['points'])}")
            print(f"   Первая точка: {route['points'][0]}")
            print(f"   Последняя точка: {route['points'][-1]}")
    else:
        print("Не найдено ни одного подходящего маршрута")
        print(f"Поместите GPX-файлы в папку {GPX_DIR}")