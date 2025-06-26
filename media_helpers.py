import requests
from urllib.parse import quote, urlencode
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import atexit
import re

# Настройки
CACHE_FILE = Path("landscape_photo_cache.json")
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

def get_photo_html(lat: float, lon: float) -> str:
    """Генерация HTML с защитой от ошибок"""
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