from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

@dataclass
class WeatherData:
    temperature: float  # in Celsius
    condition: str      # e.g., "Rain", "Snow"
    wind_speed: float   # in m/s
    precipitation_last_3days: float  # in mm (accumulated)
    precipitation_now: float  # in mm/hour (current rate)
    precipitation_forecast_max: float  # in mm/hour (max in next 8 hours)
    last_updated: datetime
    icon_url: str       # URL to weather icon
    hourly_forecast: List[Dict]  # Forecast for next 8 hours
    
    @property
    def precipitation_classification(self) -> Dict[str, bool]:
        """Classifies precipitation levels using multiple metrics"""
        return {
            # Current precipitation rate classification
            'current_rate_violent': self.precipitation_now > 50,
            'current_rate_heavy': 10 <= self.precipitation_now <= 50,
            'current_rate_moderate': 2.5 <= self.precipitation_now < 10,
            'current_rate_light': 0.1 < self.precipitation_now < 2.5,
            
            # Forecasted max precipitation rate
            'forecast_violent': self.precipitation_forecast_max > 50,
            'forecast_heavy': 10 <= self.precipitation_forecast_max <= 50,
            'forecast_moderate': 2.5 <= self.precipitation_forecast_max < 10,
            'forecast_light': 0.1 < self.precipitation_forecast_max < 2.5,
            
            # Historical precipitation (3-day accumulated)
            'historical_very_heavy': self.precipitation_last_3days > 100,
            'historical_heavy': 50 < self.precipitation_last_3days <= 100,
            'historical_moderate': 25 < self.precipitation_last_3days <= 50,
            'historical_light': 1 < self.precipitation_last_3days <= 25
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
        - Precipitation rates (current and forecast)
        - Recent precipitation history
        - Surface-specific multipliers
        """
        base_score = self.traction_score
        
        if not weather:
            return base_score
            
        # Determine weather severity based on multiple factors
        condition = 'dry'
        current_weather = weather.condition.lower()
        precip_class = weather.precipitation_classification
        
        if "snow" in current_weather or "ice" in current_weather:
            condition = 'snow'
        elif "rain" in current_weather or weather.precipitation_now > 0:
            condition = 'wet'
        
        # Get appropriate multipliers
        multipliers = self.PRECIPITATION_MULTIPLIERS[condition]
        
        # Apply reduction based on precipitation intensity
        intensity_adjustment = 1.0
        if precip_class['current_rate_violent'] or precip_class['forecast_violent']:
            intensity_adjustment = 0.6  # 40% reduction for violent rain
        elif precip_class['current_rate_heavy'] or precip_class['forecast_heavy']:
            intensity_adjustment = 0.75  # 25% reduction for heavy rain
        elif precip_class['current_rate_moderate'] or precip_class['forecast_moderate']:
            intensity_adjustment = 0.85   # 15% reduction for moderate rain
        elif precip_class['current_rate_light'] or precip_class['forecast_light']:
            intensity_adjustment = 0.95   # 5% reduction for light rain
        
        # Apply additional reduction based on precipitation history
        historical_adjustment = 1.0
        if precip_class['historical_very_heavy']:
            historical_adjustment = 0.7  # 30% reduction
        elif precip_class['historical_heavy']:
            historical_adjustment = 0.8   # 20% reduction
        elif precip_class['historical_moderate']:
            historical_adjustment = 0.9   # 10% reduction
        elif precip_class['historical_light']:
            historical_adjustment = 0.95  # 5% reduction for light accumulated rain
        
        # Calculate weighted adjustment based on surface composition
        adjusted_score = 0.0
        total_weight = 0.0
        
        for surface, proportion in self.surface_types.items():
            surface_lower = surface.lower()
            multiplier = multipliers.get(surface_lower, multipliers['default'])
            surface_weight = self.SURFACE_WEIGHTS.get(surface_lower, 0.5)
            
            # Apply all adjustments (surface-specific, intensity, and historical)
            adjusted_score += (surface_weight * multiplier * intensity_adjustment * historical_adjustment) * proportion
            total_weight += proportion
            
        if total_weight > 0:
            adjusted_score /= total_weight
            
        return min(1.0, max(0.0, adjusted_score))