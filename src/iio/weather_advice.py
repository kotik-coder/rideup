# weather_advice.py
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List, Dict
from src.iio.terrain_loader import TerrainAnalysis
from src.ui.map_helpers import print_step

@dataclass
class WeatherData:
    temperature: float  # in Celsius
    condition: str      # e.g., "Rain", "Snow"
    wind_speed: float   # in m/s
    precipitation_last_3h: float  # in mm
    precipitation_last_3days: float  # in mm
    last_updated: datetime
    icon_url: str       # URL to weather icon
    hourly_forecast: List[Dict]  # New: Forecast for next 8 hours

class WeatherAdvisor:
    @staticmethod
    def get_current_weather(bounds: List[float], api_key: str) -> Optional[WeatherData]:
        """
        Fetch current weather with 8-hour forecast
        Args:
            bounds: [min_lon, min_lat, max_lon, max_lat]
            api_key: Your OpenWeatherMap API key
        Returns:
            WeatherData object with forecast or None if failed
        """
        try:
            lat = (bounds[1] + bounds[3]) / 2
            lon = (bounds[0] + bounds[2]) / 2
            
            # Get current weather with 5-day forecast (3-hour intervals)
            url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            current = data['list'][0]
            current_weather = current['weather'][0]
            
            # Process precipitation data
            precip_3h = current.get('rain', {}).get('3h', 0) or current.get('snow', {}).get('3h', 0)
            precip_3days = precip_3h  # Include current rain in total
            
            # Get next 8 hours forecast (3 steps in 3-hour intervals)
            hourly_forecast = []
            for forecast in data['list'][1:4]:  # Next 3 forecast periods (9 hours total)
                forecast_time = datetime.fromtimestamp(forecast['dt'])
                hourly_forecast.append({
                    'time': forecast_time.strftime('%H:%M'),
                    'temp': forecast['main']['temp'],
                    'condition': forecast['weather'][0]['main'],
                    'precip': forecast.get('rain', {}).get('3h', 0) or forecast.get('snow', {}).get('3h', 0),
                    'wind': forecast['wind']['speed'],
                    'icon': f"https://openweathermap.org/img/wn/{forecast['weather'][0]['icon']}.png"
                })
                
                # Add to historical precip if in past
                if forecast['dt'] < current['dt']:
                    precip_3days += hourly_forecast[-1]['precip']

            # Debug output
            print_step("WeatherDebug", 
                f"Current: {precip_3h}mm | "
                f"3-day total: {precip_3days:.1f}mm\n"
                f"Next 8h forecast: {[f['condition'] for f in hourly_forecast]}"
            )
            
            return WeatherData(
                temperature=current['main']['temp'],
                condition=current_weather['main'],
                wind_speed=current['wind']['speed'],
                precipitation_last_3h=precip_3h,
                precipitation_last_3days=precip_3days,
                last_updated=datetime.fromtimestamp(current['dt']),
                icon_url=f"https://openweathermap.org/img/wn/{current_weather['icon']}@2x.png",
                hourly_forecast=hourly_forecast
            )
            
        except Exception as e:
            print_step("Weather", f"OpenWeatherMap error: {e}", level="ERROR")
            return None

    @staticmethod
    def generate_riding_advice(weather: WeatherData, terrain: Optional[TerrainAnalysis] = None) -> str:
        """Generate formatted riding advice without numeric values"""
        if not weather:
            return "No weather data available"
            
        advice = []
        
        # Temperature conditions
        if weather.temperature > 30:
            advice.append("🔥 Extreme heat expected - hydrate frequently and consider morning rides")
        elif weather.temperature < 0:
            advice.append("❄️ Freezing temperatures - watch for icy patches on trails")
        elif weather.temperature < 5:
            advice.append("🥶 Chilly conditions - dress in warm layers")
        
        # Current precipitation
        current_condition = weather.condition.lower()
        if "rain" in current_condition:
            advice.append("🌧️ Wet trails - expect reduced traction")
        elif "snow" in current_condition:
            advice.append("❄️ Snow on trails - possible icy sections")
        
        # Forecast conditions
        future_conditions = [f['condition'].lower() for f in weather.hourly_forecast]
        if any("rain" in cond for cond in future_conditions):
            advice.append("☔ Rain expected soon - trails may get wetter")
        if any("snow" in cond for cond in future_conditions):
            advice.append("❄️ Snow expected - prepare for icy conditions")
        
        # Terrain-specific advice
        if terrain:
            if terrain.dominant_surface == "clay":
                if "rain" in current_condition or any("rain" in f['condition'].lower() for f in weather.hourly_forecast):
                    advice.append("⚠️ Clay sections will be slippery - consider alternative routes")
                elif weather.precipitation_last_3days > 0:
                    advice.append("⚠️ Clay sections may still be damp from recent rain")
            
            if terrain.dominant_surface == "sand":
                if weather.precipitation_last_3days > 0:
                    advice.append("⚠️ Wet sand may be difficult to ride through")
            
            if terrain.traction_score < 0.4:
                advice.append("⚠️ Low traction surfaces - reduce speed on technical sections")
        
        # Wind conditions
        if weather.wind_speed > 15:
            advice.append("💨 Strong winds expected - be cautious on exposed ridges")
        elif weather.wind_speed > 10:
            advice.append("🌬️ Windy conditions - may affect balance on narrow trails")
        
        return "\n".join(advice) if advice else "✅ Ideal riding conditions - enjoy your adventure!"