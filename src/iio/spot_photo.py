# route_photo.py
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass

from routes.route import GeoPoint
from ui.map_helpers import expanded_bounds, print_step

from iio.media_helpers import photos_dir_abs
from iio.media_helpers import photos_dir_rel
from iio.media_helpers import get_exif_geolocation, get_photo_timestamp

@dataclass
class SpotPhoto:
    """
    Represents a photo associated with a route, containing its path, coordinates, and timestamp.
    """
    path: str
    fname : str
    coords: Optional[GeoPoint]
    timestamp: Optional[datetime]

    @classmethod
    def from_image_path(cls, image_path: Path) -> Optional['SpotPhoto']:
        
        if not image_path.exists:
            print_step("SpotPhoto", f"File '{image_path}' not found.", level="ERROR")
            return None    
                                
        timestamp = get_photo_timestamp(image_path)
        
        if not timestamp:  # Skip photos without timestamps
            return None  
        
        loc = get_exif_geolocation(image_path)                                          
        
        if loc:
            
            return cls(
                path=str(image_path),
                fname = os.path.basename(image_path),
                coords= GeoPoint(lon = loc.lon, lat = loc.lat),
                timestamp=timestamp                
            )        
        
        return None
        
    @staticmethod
    def load_local_photos(bounds):            
        
        #expand bounds by 5 km to filter out inappropriate photos
        exp_bounds = expanded_bounds(bounds, 5)
        lon_min, lat_min, lon_max, lat_max = exp_bounds
        
        photos = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for image_path in photos_dir_abs.glob(ext):
                photo = SpotPhoto.from_image_path(image_path)
                
                if photo and photo.coords:
                    
                    if lon_min <= photo.coords.lon and photo.coords.lon <= lon_max:
                        if lat_min < photo.coords.lat and photo.coords.lat <= lat_max:
                            photos.append(photo)
        
        print_step("SpotPhoto", f"Found {len(photos)} photos with timestamps.")
        return photos