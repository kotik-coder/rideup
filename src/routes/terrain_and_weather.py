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

    def get_wind_description(self, wind_gust: Optional[float] = None) -> str:
        """
        Convert wind speed (m/s) to descriptive text using Beaufort scale.
        Optionally considers gusts for enhanced description.
        
        Based on Beaufort Scale (knots converted to m/s):
        https://en.wikipedia.org/wiki/Beaufort_scale
        """
        if self.wind_speed < 0.5:
            desc = "Calm"
        elif self.wind_speed < 1.5:
            desc = "Light air"
        elif self.wind_speed < 3.3:
            desc = "Light breeze"
        elif self.wind_speed < 5.5:
            desc = "Gentle breeze"
        elif self.wind_speed < 7.9:
            desc = "Moderate breeze"
        elif self.wind_speed < 10.7:
            desc = "Fresh breeze"
        elif self.wind_speed < 13.8:
            desc = "Strong breeze"
        elif self.wind_speed < 17.1:
            desc = "High wind, moderate gale"
        elif self.wind_speed < 20.7:
            desc = "Gale, near gale"
        elif self.wind_speed < 24.4:
            desc = "Strong gale"
        else:
            desc = "Storm or violent storm"

        # Enhance with gust info if available
        if wind_gust and wind_gust > self.wind_speed * 1.8:  # Significant gust
            if wind_gust >= 20:
                desc += " (with severe gusts)"
            elif wind_gust >= 15:
                desc += " (with strong gusts)"
            else:
                desc += " (with gusts)"

        return desc
    
    @property
    def precipitation_classification(self) -> Dict[str, bool]:
        """Classifies precipitation levels using multiple metrics"""
        return {
            # Current precipitation rate classification
            'current_rate_violent': self.precipitation_now > 50,
            'current_rate_heavy': 10 <= self.precipitation_now <= 50,
            'current_rate_moderate': 2.5 <= self.precipitation_now < 10,
            'current_rate_light': 0.1 < self.precipitation_now < 2.5,
            'current_rate_any': self.precipitation_now > 1e-6,
            'current_rate_clear': self.precipitation_now < 1e-6,
            
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
        Calculate weather-adjusted traction score (0-1), where weather can only reduce base traction.
        """
        base_score = self.traction_score
        
        if not weather:
            return base_score  # No weather → no adjustment

        # 1. Determine current condition for multiplier selection
        current_weather = weather.condition.lower()
        precip_class = weather.precipitation_classification

        condition = 'dry'
        if "snow" in current_weather or "ice" in current_weather:
            condition = 'snow'
        elif "rain" in current_weather or weather.precipitation_now > 0:
            condition = 'wet'

        # 2. Get base surface-specific multiplier (e.g., rock stays grippy, dirt gets slippery)
        try:
            multipliers = self.PRECIPITATION_MULTIPLIERS[condition]
        except KeyError:
            multipliers = self.PRECIPITATION_MULTIPLIERS['dry']  # fallback

        # 3. Surface-weighted average of weather multipliers
        surface_multiplier = 0.0
        total_weight = 0.0

        for surface, proportion in self.surface_types.items():
            surface_lower = surface.lower()
            multiplier = multipliers.get(surface_lower, multipliers['default'])
            weight = self.SURFACE_WEIGHTS.get(surface_lower, 1.0)  # importance of surface in traction

            surface_multiplier += multiplier * weight * proportion
            total_weight += weight * proportion

        if total_weight > 0:
            surface_multiplier /= total_weight
        else:
            surface_multiplier = 1.0  # fallback

        # 4. Intensity adjustment: how bad is current/forecast rain/snow?
        intensity_multiplier = 1.0
        if precip_class['current_rate_violent'] or precip_class['forecast_violent']:
            intensity_multiplier = 0.6
        elif precip_class['current_rate_heavy'] or precip_class['forecast_heavy']:
            intensity_multiplier = 0.75
        elif precip_class['current_rate_moderate'] or precip_class['forecast_moderate']:
            intensity_multiplier = 0.85
        elif precip_class['current_rate_light'] or precip_class['forecast_light']:
            intensity_multiplier = 0.95

        # 5. Historical saturation: trails still wet from past rain?
        historical_multiplier = 1.0
        if precip_class['historical_very_heavy']:
            historical_multiplier = 0.7
        elif precip_class['historical_heavy']:
            historical_multiplier = 0.8
        elif precip_class['historical_moderate']:
            historical_multiplier = 0.9
        elif precip_class['historical_light']:
            historical_multiplier = 0.95

        # 6. Combine all multipliers (all ≤ 1.0) and apply to base traction
        total_multiplier = surface_multiplier * intensity_multiplier * historical_multiplier

        adjusted_score = base_score * total_multiplier

        # Clamp to valid range
        return max(0.0, min(1.0, adjusted_score))