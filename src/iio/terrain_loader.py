# terrain_loader.py
import osmnx as ox
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
import requests
import tempfile
import rasterio
from rasterstats import zonal_stats
from src.ui.map_helpers import print_step

@dataclass 
class TerrainAnalysis:
    surface_types: Dict[str, float]
    landcover_types: Dict[str, float] 
    dominant_surface: str
    traction_score: float
    # Simple icon mapping
    surface_icons: Dict[str, str] = field(default_factory=lambda: {
        'asphalt': '▁',  # Using simple unicode blocks
        'concrete': '▂',
        'compacted': '▃',
        'gravel': '▄',
        'dirt': '▅',
        'grass': '▆',
        'sand': '▇',
        'rock': '█'
    })
    
class TerrainLoader:
    @staticmethod
    def load_terrain(geostring: str, polygon) -> TerrainAnalysis:
        """Main entry point for terrain analysis"""
        try:
            # Get OSM surface data
            osm_surfaces = TerrainLoader._get_osm_surfaces(geostring)
            
            # Get ESA landcover data
            landcover = TerrainLoader._get_esa_landcover(polygon)
            
            # Calculate dominant surface and traction
            dominant = max(osm_surfaces.items(), key=lambda x: x[1])[0] if osm_surfaces else "unknown"
            traction = TerrainLoader._calculate_traction(osm_surfaces, landcover)
            
            return TerrainAnalysis(
                surface_types=osm_surfaces,
                landcover_types=landcover,
                dominant_surface=dominant,
                traction_score=traction
            )
            
        except Exception as e:
            print_step("TerrainLoader", f"Error in terrain analysis: {str(e)}", level="ERROR")
            return TerrainAnalysis({}, {}, "unknown", 0.5)

    @staticmethod
    def _get_osm_surfaces(geostring: str) -> Dict[str, float]:
        """Get surface types from OpenStreetMap"""
        tags = {'surface': True}
        features = ox.features_from_place(geostring, tags)
        
        if 'surface' not in features.columns:
            return {}
            
        counts = features['surface'].value_counts(normalize=True)
        return {k: v for k, v in counts.items() if v >= 0.01}  # Filter >1%

    @staticmethod
    def _get_esa_landcover(polygon):
        """Try to get ESA WorldCover data"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.tif') as tmpfile:
                TerrainLoader._download_esa_tile(polygon.bounds, tmpfile.name)
                
                with rasterio.open(tmpfile.name) as src:
                    stats = zonal_stats(
                        polygon,
                        src.read(1),
                        affine=src.transform,
                        categorical=True,
                        category_map=ESA_LANDCOVER_MAP
                    )
            return stats[0] if stats else {}
        except Exception:
            print_step("TerrainLoader", "ESA WorldCover unavailable", level="WARNING")
            return {}

    @staticmethod
    def _download_esa_tile(bounds, output_path):
        """Download ESA WorldCover tile"""
        lat = (bounds[1] + bounds[3]) / 2
        lon = (bounds[0] + bounds[2]) / 2
        tile_url = f"https://esa-worldcover.s3.amazonaws.com/v100/2020/ESA_WorldCover_10m_{lat}_{lon}.tif"
        
        response = requests.get(tile_url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    @staticmethod
    def _calculate_traction(surfaces: Dict[str, float], landcover: Dict[str, float]) -> float:
        """Calculate traction score (0-1)"""
        SURFACE_WEIGHTS = {
            'asphalt': 0.9, 'concrete': 0.85, 'compacted': 0.8,
            'fine_gravel': 0.7, 'gravel': 0.6, 'dirt': 0.5,
            'grass': 0.4, 'sand': 0.3, 'mud': 0.1
        }
        
        if not surfaces:
            return 0.5
            
        total_weight = sum(SURFACE_WEIGHTS.get(k.lower(), 0.5) * v 
                       for k, v in surfaces.items())
        return min(1.0, max(0.0, total_weight))

# Constants
ESA_LANDCOVER_MAP = {
    10: "Trees", 20: "Shrubland", 30: "Grassland",
    40: "Cropland", 50: "Built-up", 60: "Bare",
    70: "Snow/Ice", 80: "Water", 90: "Wetlands"
}