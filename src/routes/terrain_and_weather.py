from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

@dataclass
class WeatherData:
    temperature: float  # in Celsius
    condition: str      # e.g., "Rain", "Snow"
    wind_speed: float   # in m/s
    precipitation_last_3days: float  # in mm
    precipitation_now: float  # in mm
    last_updated: datetime
    icon_url: str       # URL to weather icon
    hourly_forecast: List[Dict]  # Forecast for next 8 hours
    
    @property
    def precipitation_classification(self) -> Dict[str, bool]:
        """Classifies precipitation levels for European Russia"""
        return {
            'very_heavy': self.precipitation_last_3days > 40,  # >40mm in 3 days
            'heavy': 25 < self.precipitation_last_3days <= 40,  # 25-40mm
            'moderate': 15 < self.precipitation_last_3days <= 25,  # 15-25mm
            'some': 5 < self.precipitation_last_3days <= 15,  # 5-15mm
            'light': self.precipitation_last_3days <= 5  # <5mm
        }

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

    # Precipitation impact multipliers by surface type
    PRECIPITATION_MULTIPLIERS = {
        # Dry conditions (no multiplier)
        'dry': {
            'default': 1.0
        },
        # Wet conditions (rain)
        'wet': {
            'asphalt': 0.85, 'concrete': 0.8, 'paved': 0.8,
            'cobblestone': 0.5, 'compacted': 0.6, 'fine_gravel': 0.55,
            'gravel': 0.5, 'ground': 0.4, 'pebblestone': 0.45,
            'dirt': 0.3, 'grass': 0.25, 'sand': 0.4, 'mud': 0.1,
            'wood': 0.2, 'rock': 0.3, 'metal': 0.15,
            'default': 0.35
        },
        # Snow/ice conditions
        'snow': {
            'asphalt': 0.3, 'concrete': 0.25, 'paved': 0.3,
            'cobblestone': 0.2, 'compacted': 0.35, 'fine_gravel': 0.4,
            'gravel': 0.45, 'ground': 0.5, 'pebblestone': 0.4,
            'dirt': 0.3, 'grass': 0.35, 'sand': 0.6,
            'wood': 0.15, 'rock': 0.2, 'metal': 0.1,
            'default': 0.25
        }
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
        
    def get_adjusted_traction(self, weather: Optional[WeatherData] = None) -> float:
        """
        Calculate weather-adjusted traction score (0-1) accounting for:
        - Current weather conditions
        - Recent precipitation history
        - Surface-specific multipliers
        """
        base_score = self.traction_score
        
        if not weather:
            return base_score
            
        # Determine weather severity based on current and historical precipitation
        condition = 'dry'
        current_weather = weather.condition.lower()
        precip_class = weather.precipitation_classification
        
        if "snow" in current_weather or "ice" in current_weather:
            condition = 'snow'
        elif "rain" in current_weather or weather.precipitation_now > 0:
            condition = 'wet'
        
        # Get appropriate multipliers
        multipliers = self.PRECIPITATION_MULTIPLIERS[condition]
        
        # Apply additional reduction based on precipitation history
        historical_adjustment = 1.0
        if precip_class['very_heavy']:
            historical_adjustment = 0.7  # 30% reduction for very heavy rain
        elif precip_class['heavy']:
            historical_adjustment = 0.8   # 20% reduction
        elif precip_class['moderate']:
            historical_adjustment = 0.9   # 10% reduction
        
        # Calculate weighted adjustment based on surface composition
        adjusted_score = 0.0
        total_weight = 0.0
        
        for surface, proportion in self.surface_types.items():
            surface_lower = surface.lower()
            multiplier = multipliers.get(surface_lower, multipliers['default'])
            surface_weight = self.SURFACE_WEIGHTS.get(surface_lower, 0.5)
            
            # Apply both surface-specific and historical adjustments
            adjusted_score += (surface_weight * multiplier * historical_adjustment) * proportion
            total_weight += proportion
            
        if total_weight > 0:
            adjusted_score /= total_weight
            
        return min(1.0, max(0.0, adjusted_score))