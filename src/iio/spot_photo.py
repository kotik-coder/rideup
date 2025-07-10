# route_photo.py
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class SpotPhoto:
    """
    Represents a photo associated with a route, containing its path, coordinates, and timestamp.
    """
    path: str
    coords: Optional[Tuple[float, float]]
    timestamp: Optional[datetime]

    @classmethod
    def from_image_path(cls, image_path: Path) -> Optional['SpotPhoto']:
        """
        Creates a RoutePhoto instance from an image path by extracting EXIF data.
        Returns None if the image cannot be processed.
        """
        try:
            from media_helpers import get_exif_geolocation, get_photo_timestamp
            
            timestamp = get_photo_timestamp(image_path)
            if not timestamp:  # Skip photos without timestamps
                return None
                
            return cls(
                path=str(image_path),
                coords=get_exif_geolocation(image_path),
                timestamp=timestamp
            )
        except Exception:
            return None