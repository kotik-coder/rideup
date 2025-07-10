import os
import requests
from urllib.parse import quote, urlencode
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import atexit
from datetime import datetime, timezone
from routes.route import GeoPoint

# Импортируем Pillow для обработки изображений и piexif для EXIF
from PIL import Image
import piexif

# Get the package root directory
package_root = Path(__file__).parent.parent.parent
photos_dir_rel = "local_photos" 
photos_dir_abs = package_root / photos_dir_rel
CACHE_FILE = Path(package_root / "landscape_photo_cache.json")

# Настройки
TIMEOUT = 15
USER_AGENT = "ReliablePhotoLoader/1.0"
SEARCH_RADIUS = 200  # 200 метров

# Расширенный список ключевых слов для ландшафтов
LANDSCAPE_KEYWORDS = [
    'landscape', 'nature', 'view', 'panorama', 'scenery',
    'forest', 'wood', 'tree', 'park', 'wilderness',
    'mountain', 'hill', 'valley', 'ridge', 'cliff',
    'river', 'lake', 'stream', 'waterfall', 'spring',
    'field', 'meadow', 'trail', 'path', 'hiking',
    'пейзаж', 'природа', 'вид', 'лес', 'гора',
    'река', 'озеро', 'поле', 'тропа', 'ландшафт'
]

class PhotoCache:
    def __init__(self):
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, dict]:
        """Загружает кэш с проверкой структуры"""
        try:
            if CACHE_FILE.exists():
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
        except Exception:
            return {}
        return {}

    def save(self):
        """Сохраняет кэш с обработкой ошибок"""
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f)
        except Exception:
            pass

    def get(self, key: str) -> Optional[dict]:
        """Безопасное получение данных из кэша"""
        return self.cache.get(key)

    def set(self, key: str, photo_data: dict):
        """Безопасное сохранение данных"""
        if isinstance(photo_data, dict):
            self.cache[key] = photo_data

cache = PhotoCache()

def _convert_to_degrees(value: tuple) -> float:
    """
    Конвертирует значение GPS из формата EXIF (градусы, минуты, секунды) в десятичные градусы.
    Пример: ((35, 1), (30, 1), (15, 1)) -> 35.50416666666666
    """
    d = float(value[0][0]) / float(value[0][1])
    m = float(value[1][0]) / float(value[1][1])
    s = float(value[2][0]) / float(value[2][1])
    return d + (m / 60.0) + (s / 3600.0)

def get_exif_geolocation(image_path: Path) -> Optional[GeoPoint]:
    """
    Извлекает географические координаты (широту, долготу) из EXIF данных фотографии.
    Возвращает (latitude, longitude) или None, если данные не найдены.
    """
    try:
        img = Image.open(image_path)
        exif_dict = piexif.load(img.info["exif"])

        if piexif.GPSIFD.GPSLatitude in exif_dict["GPS"] and \
           piexif.GPSIFD.GPSLongitude in exif_dict["GPS"] and \
           piexif.GPSIFD.GPSLatitudeRef in exif_dict["GPS"] and \
           piexif.GPSIFD.GPSLongitudeRef in exif_dict["GPS"]:
            
            lat_data = exif_dict["GPS"][piexif.GPSIFD.GPSLatitude]
            lat_ref = exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef].decode('utf-8')
            lon_data = exif_dict["GPS"][piexif.GPSIFD.GPSLongitude]
            lon_ref = exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef].decode('utf-8')

            latitude = _convert_to_degrees(lat_data)
            if lat_ref != "N":
                latitude = -latitude

            longitude = _convert_to_degrees(lon_data)
            if lon_ref != "E":
                longitude = -longitude
            
            return GeoPoint(lat = latitude, lon = longitude)
    except Exception as e:
        pass
    return None

def get_photo_timestamp(image_path: Path) -> Optional[datetime]:
    """Извлекает временную метку создания фото из EXIF данных и конвертирует в UTC."""
    try:
        img = Image.open(image_path)
        exif_dict = piexif.load(img.info.get('exif', b''))
        
        # Check ExifIFD first (where DateTimeOriginal usually is)
        if piexif.ExifIFD.DateTimeOriginal in exif_dict.get('Exif', {}):
            dt_str = exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal].decode('utf-8')
        elif piexif.ExifIFD.DateTimeDigitized in exif_dict.get('Exif', {}):
            dt_str = exif_dict['Exif'][piexif.ExifIFD.DateTimeDigitized].decode('utf-8')
        elif piexif.ImageIFD.DateTime in exif_dict.get('0th', {}):
            dt_str = exif_dict['0th'][piexif.ImageIFD.DateTime].decode('utf-8')
        else:
            return None

        # EXIF timestamp format is "YYYY:MM:DD HH:MM:SS" (local time)
        dt_obj = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        
        # Assume the timestamp is in local time and convert to UTC
        # Note: This assumes the local timezone matches the system timezone
        # For more precise handling, you might need to get the timezone from EXIF
        return dt_obj.astimezone(timezone.utc)
        
    except Exception:
        return None

def get_photos_with_geolocation_from_folder(folder_path: Path) -> List[Dict[str, any]]:
    """
    Сканирует указанную папку на наличие изображений (jpg, jpeg, png)
    и извлекает из них географические координаты EXIF и временные метки.
    Возвращает список словарей, каждый из которых содержит:
    - 'path': путь к файлу
    - 'coords': (latitude, longitude) или None
    - 'timestamp': datetime объекта или None
    """
    photos_with_metadata = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for image_path in folder_path.glob(ext):
            try:
                photos_with_metadata.append({
                    'path': str(image_path),
                    'coords': get_exif_geolocation(image_path),
                    'timestamp': get_photo_timestamp(image_path)
                })
            except Exception:
                continue
    return photos_with_metadata


def get_landscape_photo(lat: float, lon: float) -> Tuple[Optional[str], str]:
    """Основная функция для получения фото"""
    cache_key = f"{lat:.5f},{lon:.5f}"
    
    # 1. Проверка кэша с защитой от None
    cached = cache.get(cache_key)
    if cached and isinstance(cached, dict):
        return cached.get('url'), cached.get('source', '')
    
    # 2. Поиск фото
    photo_data = _find_landscape_photo(lat, lon)
    
    # 3. Fallback на Яндекс если нужно
    if not photo_data.get('url'):
        photo_data = {
            'url': _get_yandex_satellite(lat, lon),
            'source': 'Яндекс.Спутник'
        }
    
    # 4. Сохраняем в кэш
    cache.set(cache_key, photo_data)
    return photo_data.get('url'), photo_data.get('source', '')

def _find_landscape_photo(lat: float, lon: float) -> dict:
    """Поиск ландшафтного фото с защитой от ошибок"""
    result = {'url': None, 'source': 'Wikimedia Commons'}
    
    try:
        # 1. Поиск в Wikimedia
        search_url = (
            "https://commons.wikimedia.org/w/api.php?"
            "action=query&"
            "list=geosearch&"
            f"gscoord={lat}|{lon}&"
            f"gsradius={SEARCH_RADIUS}&"
            "gslimit=20&"
            "gsnamespace=6&"
            "format=json"
        )
        
        response = requests.get(search_url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
        if response.status_code != 200:
            return result
            
        data = response.json()
        query = data.get('query', {})
        geosearch = query.get('geosearch', []) if isinstance(query, dict) else []
        
        # 2. Фильтрация по ключевым словам
        for item in geosearch:
            if not isinstance(item, dict):
                continue
                
            title = item.get('title', '').lower()
            if (any(kw in title for kw in LANDSCAPE_KEYWORDS) and \
               title.endswith(('.jpg', '.jpeg', '.png'))):
                
                # 3. Получение URL фото
                image_url = (
                    "https://commons.wikimedia.org/w/api.php?"
                    "action=query&"
                    "prop=imageinfo&"
                    "iiprop=url&"
                    f"titles={quote(item['title'])}&"
                    "iiurlwidth=800&"
                    "format=json"
                )
                
                img_response = requests.get(image_url, timeout=TIMEOUT)
                if img_response.status_code == 200:
                    img_data = img_response.json()
                    pages = img_data.get('query', {}).get('pages', {})
                    if pages:
                        page = next(iter(pages.values()))
                        if isinstance(page, dict):
                            result['url'] = page.get('imageinfo', [{}])[0].get('url')
                            break
    except Exception:
        pass
        
    return result

def _get_yandex_satellite(lat: float, lon: float) -> str:
    """Генерация URL спутникового снимка"""
    params = {
        'll': f"{lon},{lat}",
        'z': '16',
        'l': 'sat',
        'size': '650,450',
        'pt': f"{lon},{lat},pm2rdl"
    }
    return f"https://static-maps.yandex.ru/1.x/?{urlencode(params)}"

def get_photo_html(lat: float, lon: float, local_photo_path: Optional[str] = None) -> str:
    """Генерация HTML с защитой от ошибок"""
    photo_url = None
    source = ""
    
    if local_photo_path:
        photo_url = f"/static/local_photos/{local_photo_path}"
        source = "Локальное фото"
    else:
        photo_url, source = get_landscape_photo(lat, lon)
    
    if not photo_url:
        return """
        <div style="margin:10px 0;text-align:center;color:#666;font-size:0.9em">
            <p>Фото местности недоступно</p>
        </div>
        """        
    
    return f"""
    <div style="margin:10px 0;text-align:center">
        <img src="{photo_url}" 
             onerror="this.src='{_get_yandex_satellite(lat, lon)}'"
             style="max-width:100%;max-height:200px;border-radius:4px;border:1px solid #eee;">
        <p style="font-size:0.8em;color:#999;margin-top:5px">
            Источник: {source}
        </p>
    </div>
    """

# Сохранение кэша при выходе
atexit.register(cache.save)