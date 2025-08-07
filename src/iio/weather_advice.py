# weather_advice.py
import requests
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional, List, Dict
from src.iio.terrain_loader import TerrainAnalysis
from src.ui.map_helpers import print_step
from src.routes.terrain_and_weather import WeatherData

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
            
            # Current precipitation (now)
            precip_now = current.get('rain', {}).get('1h', 0) or current.get('snow', {}).get('1h', 0)  # Using 1h as "now"
            
            # Calculate precipitation over available historical period (3 days)
            precip_3days = 0
            cutoff_time = datetime.now() - timedelta(days=3)
            
            for forecast in data['list']:
                forecast_time = datetime.fromtimestamp(forecast['dt'])
                if forecast_time < cutoff_time:
                    continue
                precip = forecast.get('rain', {}).get('3h', 0) or forecast.get('snow', {}).get('3h', 0)
                precip_3days += precip
            
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

            # Debug output
            print_step("WeatherDebug", 
                f"Current: {precip_now}mm | "
                f"3-day total: {precip_3days:.1f}mm\n"
                f"Next 8h forecast: {[f['condition'] for f in hourly_forecast]}"
            )
            
            return WeatherData(
                temperature=current['main']['temp'],
                condition=current_weather['main'],
                wind_speed=current['wind']['speed'],
                precipitation_now=precip_now,
                precipitation_last_3days=precip_3days,
                last_updated = datetime.now(timezone.utc),
                icon_url=f"https://openweathermap.org/img/wn/{current_weather['icon']}@2x.png",
                hourly_forecast=hourly_forecast
            )
            
        except Exception as e:
            print_step("Weather", f"OpenWeatherMap error: {e}", level="ERROR")
            return None

    @staticmethod
    def generate_riding_advice(weather: WeatherData, terrain: Optional[TerrainAnalysis] = None) -> str:
        """Generate formatted riding advice using precipitation classification"""
        if not weather:
            return "No weather data available"
            
        advice = []
        precip_class = weather.precipitation_classification
        
        # Temperature conditions
        if weather.temperature > 30:
            advice.append("🔥 Extreme heat expected - hydrate frequently and consider morning rides")
        elif weather.temperature < 0:
            advice.append("❄️ Freezing temperatures - watch for icy patches on trails")
        elif weather.temperature < 5:
            advice.append("🥶 Chilly conditions - dress in warm layers")
        
        # Current precipitation analysis
        current_condition = weather.condition.lower()
        if "rain" in current_condition:
            if precip_class['very_heavy']:
                advice.append("🌧️ Torrential rain - trails will be severely affected")
            elif precip_class['heavy']:
                advice.append("🌧️ Heavy rain - significant trail deterioration expected")
            else:
                advice.append("🌧️ Light rain - trails becoming wet")
        elif "snow" in current_condition:
            advice.append("❄️ Snow on trails - possible icy sections")
        
        # Historical precipitation impact
        if precip_class['very_heavy']:
            advice.append("⚠️ Exceptional rainfall - many trails likely impassable")
        elif precip_class['heavy']:
            advice.append("⚠️ Heavy recent rainfall - trails saturated and muddy")
        elif precip_class['moderate']:
            advice.append("⚠️ Considerable recent rain - many wet sections")
        elif precip_class['some']:
            advice.append("⚠️ Some recent rain - patches of mud possible")
        
        # Forecast conditions
        future_conditions = [f['condition'].lower() for f in weather.hourly_forecast]
        if any("rain" in cond for cond in future_conditions):
            if precip_class['very_heavy'] or precip_class['heavy']:
                advice.append("☔ Additional rain coming - already poor conditions will worsen")
            elif precip_class['moderate']:
                advice.append("☔ More rain expected - trail conditions deteriorating")
            else:
                advice.append("☔ Rain expected soon - trails may get wetter")
        
        if any("snow" in cond for cond in future_conditions):
            advice.append("❄️ Snow expected - prepare for icy conditions")
                
        # Terrain-specific advice
        if terrain:
            # Get weather-adjusted traction score
            adjusted_traction = terrain.get_adjusted_traction(weather)
            
            # Clay-specific advice
            if terrain.dominant_surface == "clay":
                if precip_class['very_heavy']:
                    advice.append("⛔ Clay trails impassable after this much rain")
                elif precip_class['heavy']:
                    advice.append("⚠️ Clay sections extremely slippery - avoid completely")
                elif precip_class['moderate'] or "rain" in current_condition:
                    advice.append("⚠️ Clay becomes treacherous when wet - use extreme caution")
                elif precip_class['some']:
                    advice.append("⚠️ Clay sections may remain slippery")
                # Additional warning based on adjusted traction
                if adjusted_traction < 0.3:
                    advice.append("🚨 Critical clay conditions - extremely low traction")
            
            # Sand-specific advice
            if terrain.dominant_surface == "sand":
                if "snow" in current_condition:
                    advice.append("⚠️ Snow on sand - traction may improve but watch for hidden obstacles")
                elif precip_class['heavy'] or precip_class['very_heavy']:
                    advice.append("⚠️ Wet sand will be heavy and difficult to ride through")
                elif precip_class['moderate']:
                    advice.append("⚠️ Sand may be compacted and slow-rolling")
                # Additional advice based on adjusted traction
                if adjusted_traction < 0.35:
                    advice.append("⚠️ Sandy sections require significant effort")
            
            # General traction advice using adjusted score
            if adjusted_traction < 0.3:
                advice.append("⛔ Dangerously low traction - riding not recommended")
            elif adjusted_traction < 0.4:
                advice.append("⚠️ Very poor traction - experts only with proper equipment")
            elif adjusted_traction < 0.5:
                advice.append("⚠️ Reduced traction - ride cautiously")
            
            # Surface-specific traction warnings
            if adjusted_traction < 0.5:
                if terrain.dominant_surface == "rock":
                    advice.append("⚠️ Slippery rock surfaces - maintain low speed")
                elif terrain.dominant_surface == "wood":
                    advice.append("⚠️ Wooden features extremely slippery when wet")
                elif terrain.dominant_surface == "metal":
                    advice.append("⚠️ Metal surfaces become dangerously slick when wet")
        
        # Wind conditions
        if weather.wind_speed > 15:
            advice.append("💨 Strong winds expected - be cautious on exposed ridges")
        elif weather.wind_speed > 10:
            advice.append("🌬️ Windy conditions - may affect balance on narrow trails")
        
        return "\n".join(advice) if advice else "✅ Good riding conditions - enjoy your adventure!"