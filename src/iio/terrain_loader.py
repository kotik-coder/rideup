# terrain_loader.py
import osmnx as ox
from typing import Dict
from dataclasses import dataclass
from src.ui.map_helpers import print_step

@dataclass 
class TerrainAnalysis:
    surface_types: Dict[str, float]
    dominant_surface: str
    traction_score: float
    
    # Surface weights and icons remain the same
    SURFACE_WEIGHTS = {
        # High-traction
        'asphalt': 0.9, 'concrete': 0.85, 'paved': 0.8, 'cobblestone': 0.75,
        # Medium-traction
        'compacted': 0.7, 'fine_gravel': 0.65, 'gravel': 0.6, 
        'ground': 0.55, 'pebblestone': 0.5,
        # Low-traction
        'dirt': 0.45, 'grass': 0.4, 'sand': 0.3, 'mud': 0.2, 'wood': 0.35,
        # Special cases
        'rock': 0.4, 'metal': 0.3, 'snow': 0.1
    }

    surface_icons = {
        # High-traction
        'asphalt': '▁', 'concrete': '▂', 'paved': '▂', 'cobblestone': '▃',
        # Medium-traction
        'compacted': '▃', 'fine_gravel': '▄', 'gravel': '▄', 
        'ground': '▅', 'pebblestone': '▅',
        # Low-traction
        'dirt': '▅', 'grass': '▆', 'sand': '▇', 'mud': '▇', 'wood': '▆',
        # Special cases
        'rock': '█', 'metal': '▂', 'snow': '❄️'
    }
    
class TerrainLoader:
    @staticmethod
    def load_terrain(geostring: str, polygon = None) -> TerrainAnalysis:
        """Main entry point for terrain analysis (polygon now optional)"""
        try:
            # Get enhanced OSM surface data (including Russia-specific tags)
            osm_surfaces = TerrainLoader._get_osm_surfaces(geostring)
            
            # Calculate dominant surface and traction
            dominant = max(osm_surfaces.items(), key=lambda x: x[1])[0] if osm_surfaces else "unknown"
            traction = TerrainLoader._calculate_traction(osm_surfaces)
            
            return TerrainAnalysis(
                surface_types=osm_surfaces,
                dominant_surface=dominant,
                traction_score=traction
            )
            
        except Exception as e:
            print_step("TerrainLoader", f"Error in terrain analysis: {str(e)}", level="ERROR")
            return TerrainAnalysis({}, "unknown", 0.5)

    @staticmethod
    def _get_osm_surfaces(geostring: str) -> Dict[str, float]:
        """Get enhanced surface types from OpenStreetMap with Russia-specific tags"""
        tags = {
            'surface': True,
            'tracktype': True,  # Common in rural Russia (grades 1-5)
            'highway': True      # For road types that imply surface
        }
        
        try:
            features = ox.features_from_place(geostring, tags)
            
            # Normalize surface types (merge similar tags)
            if 'surface' in features.columns:
                # Map Russian-specific tags to standard ones
                surface_map = {
                    'грунт': 'ground',         # Russian for "ground"
                    'песок': 'sand',           # Russian for "sand"
                    'асфальт': 'asphalt',      # Russian for "asphalt"
                    'земля': 'dirt',           # Russian for "dirt/earth"
                    'tracktype_1': 'compacted',
                    'tracktype_2': 'gravel',
                    'tracktype_3': 'dirt',
                }
                
                features['surface'] = features['surface'].str.lower().replace(surface_map)
                counts = features['surface'].value_counts(normalize=True)
                return {k: v for k, v in counts.items() if v >= 0.01}
                
            return {}
            
        except Exception as e:
            print_step("TerrainLoader", f"OSM query failed: {str(e)}", level="WARNING")
            return {}

    @staticmethod
    def _calculate_traction(surfaces: Dict[str, float]) -> float:
        """Calculate traction score (0-1) based on surface types"""
        if not surfaces:
            return 0.5
            
        total_weight = sum(TerrainAnalysis.SURFACE_WEIGHTS.get(k.lower(), 0.5) * v 
                       for k, v in surfaces.items())
        return min(1.0, max(0.0, total_weight))

    # Optional: Add Russian data source integration later
    @staticmethod
    def _get_ru_terrain_data(polygon):
        """Placeholder for Russian-specific terrain data (e.g., Rosreestr)"""
        # Potential sources:
        # 1. Rosreestr API (https://rosreestr.gov.ru)
        # 2. OpenStreetMap Russia special tags
        # 3. Yandex Maps API (paid)
        pass