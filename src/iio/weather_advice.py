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
        Fetch current weather, 3-day precipitation history (if possible), and 8-hour forecast.
        
        Args:
            bounds: [min_lon, min_lat, max_lon, max_lat]
            api_key: Your OpenWeatherMap API key
            
        Returns:
            WeatherData object or None if failed
        """
        try:
            # Center of bounding box
            lat = (bounds[1] + bounds[3]) / 2
            lon = (bounds[0] + bounds[2]) / 2

            # === 1. Get CURRENT weather ===
            current_url = (
                f"http://api.openweathermap.org/data/2.5/weather"
                f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            )
            current_resp = requests.get(current_url, timeout=10)
            current_resp.raise_for_status()
            current_data = current_resp.json()

            # === 2. Get FORECAST (next 9 hours) ===
            forecast_url = (
                f"http://api.openweathermap.org/data/2.5/forecast"
                f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            )
            forecast_resp = requests.get(forecast_url, timeout=10)
            forecast_resp.raise_for_status()
            forecast_data = forecast_resp.json()

            current_main = current_data['main']
            current_weather = current_data['weather'][0]
            current_wind = current_data['wind']

            # Current precipitation (only if available)
            precip_now = (
                current_data.get('rain', {}).get('1h', 0) or 
                current_data.get('snow', {}).get('1h', 0)
            )

            # === 3. Forecast for next ~9 hours (3 intervals) ===
            near_forecast = forecast_data['list'][:3]  # Next 3 periods (~9 hours)
            hourly_forecast = []
            precip_forecast_values = []

            for item in near_forecast:
                item_time = datetime.fromtimestamp(item['dt'])
                rain_3h = item.get('rain', {}).get('3h', 0)
                snow_3h = item.get('snow', {}).get('3h', 0)
                total_precip = rain_3h + snow_3h  # mm over 3 hours
                precip_rate = total_precip / 3  # mm/hour average

                precip_forecast_values.append(precip_rate)

                hourly_forecast.append({
                    'time': item_time.strftime('%H:%M'),
                    'temp': item['main']['temp'],
                    'condition': item['weather'][0]['main'],
                    'precip': total_precip,
                    'wind': item['wind']['speed'],
                    'icon': f"https://openweathermap.org/img/wn/{item['weather'][0]['icon']}.png"
                })

            precip_forecast_max = max(precip_forecast_values) if precip_forecast_values else 0

            # === 4. Historical precipitation: NOT AVAILABLE via standard API ===
            # OpenWeatherMap does not provide past 3 days via /forecast or /weather
            # You need One Call API 3.0 (paid tier) or timemachine endpoint
            # For now, set to negative value with warning
            print_step("Weather", "Historical precipitation not available with current API endpoints.", level="WARNING")
            precip_3days = -0.1  # Cannot fetch without One Call API

            # Debug output
            print_step("WeatherDebug", 
                f"Location: {lat:.4f}, {lon:.4f} | "
                f"Current: {precip_now:.1f}mm/h | "
                f"Max forecast (next 9h): {precip_forecast_max:.1f}mm/h | "
                f"Conditions: {[f['condition'] for f in hourly_forecast]}"
            )

            return WeatherData(
                temperature=current_main['temp'],
                condition=current_weather['main'],
                wind_speed=current_wind.get('speed', 0),
                precipitation_now=precip_now,
                precipitation_last_3days=precip_3days,
                precipitation_forecast_max=precip_forecast_max,
                last_updated=datetime.now(timezone.utc),
                icon_url=f"https://openweathermap.org/img/wn/{current_weather['icon']}@2x.png",
                hourly_forecast=hourly_forecast
            )

        except requests.exceptions.RequestException as e:
            print_step("Weather", f"HTTP error fetching weather: {e}", level="ERROR")
            return None
        except KeyError as e:
            print_step("Weather", f"Missing expected data field: {e}", level="ERROR")
            return None
        except Exception as e:
            print_step("Weather", f"Unexpected error: {e}", level="ERROR")
            return None

    @staticmethod
    def generate_riding_advice(weather: WeatherData, terrain: Optional[TerrainAnalysis] = None) -> str:
        """Generate formatted riding advice using precipitation classification"""
        if not weather:
            return "No weather data available"
            
        advice = []
        precip_class = weather.precipitation_classification
        
        # Temperature conditions (unchanged)
        if weather.temperature > 30:
            advice.append("🔥 Extreme heat (>30°C) – high dehydration risk. Carry extra water and ride early.")
            advice.append("🛑 Expect soft trail surfaces – mud, dust, or tacky clay may affect grip and braking.")
        elif weather.temperature < 0:
            advice.append("❄️ Freezing temps – ice likely on shaded trails, rocks, and roots. Ride with caution.")
            advice.append("🧤 Consider studded tires or microspikes for control on icy descents.")
        elif weather.temperature < 5:
            advice.append("🥶 Cold weather (<5°C) – warm layers essential. Brake performance may drop in cold.")
            advice.append("🧯 Watch for black ice in shaded areas, especially in morning/evening rides.")
        elif weather.temperature < 15:
            advice.append("🧥 Cool riding conditions – ideal for endurance, but bring a windproof jacket.")
        
        # Current precipitation analysis - updated to use new classification
        current_condition = weather.condition.lower()
        if "rain" in current_condition or precip_class['current_rate_any']:
            if precip_class['current_rate_violent']:
                advice.append("🌧️ Torrential rain (>50mm/h) – immediate danger! Trails turning to mud. Seek shelter.")
                advice.append("🛑 Risk of erosion and washouts – avoid trail use to protect environment.")
            elif precip_class['current_rate_heavy']:
                advice.append("🌧️ Heavy rain (10–50mm/h) – rapid trail degradation. Expect deep mud and poor drainage.")
                advice.append("⚠️ Braking distance increased – ride slower, avoid skidding.")
            elif precip_class['current_rate_moderate']:
                advice.append("🌧️ Moderate rain (2.5–10mm/h) – trails getting wet. Avoid soft sections to prevent ruts.")
            else:
                advice.append("🌦️ Light rain (<2.5mm/h) – minor impact. Trails damp but ridable.")
        elif "snow" in current_condition:
            advice.append("❄️ Snow falling – trails accumulating snow. Icy patches likely under fresh snow.")
            advice.append("🛑 Traction severely reduced – only ride with aggressive tires or studs.")
        
        # Historical precipitation impact - updated thresholds
        if precip_class['historical_very_heavy']:  # >100mm
            advice.append("⚠️ Extreme saturation (>100mm/3d) – most trails impassable. Deep mud, pooling water.")
            advice.append("🛑 Riding now causes permanent trail damage. Strongly advise postponing.")
        elif precip_class['historical_heavy']:  # 50–100mm
            advice.append("⚠️ Heavy recent rain (50–100mm/3d) – trails saturated. Expect slow, muddy conditions.")
            advice.append("🥾 Ride only if necessary – avoid climbs and switchbacks to prevent rutting.")
        elif precip_class['historical_moderate']:  # 25–50mm
            advice.append("⚠️ Moderate rain (25–50mm/3d) – many wet sections. Drainage may be overwhelmed.")
            advice.append("🛣️ Some trails sticky – choose well-drained or gravel-accessed routes.")
        elif precip_class['historical_light']:  # <25mm
            advice.append("🟢 Light recent rain (<25mm/3d) – mostly rideable. Some damp patches, dries quickly.")
        
        # Forecast conditions - updated to use forecast classification
        if precip_class['forecast_violent']:
            advice.append("☔ Torrential rain forecast (>50mm/h) – dangerous conditions incoming. Delay ride.")
        elif precip_class['forecast_heavy']:
            advice.append("☔ Heavy rain expected (10–50mm/h) – trail conditions will rapidly worsen.")
            advice.append("⏱️ If already riding, exit exposed areas soon. Avoid long descents in deepening mud.")
        elif precip_class['forecast_moderate']:
            advice.append("☔ Moderate rain coming (2.5–10mm/h) – trails will get wet. Start early to beat rain.")
        elif precip_class['forecast_light']:
            advice.append("🌦️ Light rain expected (<2.5mm/h) – minimal impact. Ride as planned with light rain gear.")

        # Snow in forecast
        if any("snow" in f['condition'].lower() for f in weather.hourly_forecast):
            advice.append("❄️ Snow in forecast – icy or snow-covered trails expected. Check trail reports.")
            advice.append("🛑 Cold + snow = high fall risk. Only ride if prepared for winter conditions.")

        total_risk = sum(1 for a in advice if any(bad in a for bad in ["⚠️", "🔥", "🌧️", "☔", "❄️"])) 

        if total_risk >= 4:
            advice.append("🔴 Strongly consider postponing or choosing an alternative route.")
        elif total_risk >= 2:
            advice.append("🟡 Ride with caution and proper gear.")
                
        # Terrain-specific advice - updated to use new classification
        if terrain:
            adjusted_traction = terrain.get_adjusted_traction(weather)
            
            # Clay-specific advice
            if terrain.dominant_surface == "clay":
                if precip_class['current_rate_heavy'] or precip_class['current_rate_violent']:
                    advice.append("⛔ Clay trails impassable in heavy rain")
                elif precip_class['current_rate_moderate']:
                    advice.append("⚠️ Clay becomes treacherous in moderate rain")
                elif precip_class['historical_heavy'] or precip_class['historical_very_heavy']:
                    advice.append("⚠️ Clay remains dangerous after heavy rainfall")
            
            # Sand-specific advice
            if terrain.dominant_surface == "sand":
                if "snow" in current_condition:
                    advice.append("⚠️ Snow on sand - traction may improve but watch for hidden obstacles")
                elif precip_class['current_rate_heavy'] or precip_class['current_rate_violent']:
                    advice.append("⚠️ Wet sand will be heavy and difficult to ride through")
            
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
            
        if weather.wind_speed >= 20:
            advice.append("⚠️ 🚫 Storm-force winds! Extremely dangerous for MTB.")
            advice.append("🛑 Avoid exposed ridges, cliffside trails, and open summits – risk of being blown off balance.")
            advice.append("🪵 High chance of downed branches – avoid forested singletrack after storms.")
            advice.append("💡 Consider postponing your ride – control and safety are severely compromised.")

        elif weather.wind_speed >= 15:
            advice.append("💨 Strong winds – expect challenging handling on exposed sections.")
            advice.append("📉 Be ready for sudden gusts that can push you sideways on narrow trails or berms.")
            advice.append("🛡️ Tuck low on descents to reduce wind resistance and improve stability.")

        elif weather.wind_speed >= 10:
            advice.append("🌬️ Windy conditions – may affect balance at speed or on open fire roads.")
            advice.append("🧭 Stay centered on the bike and anticipate crosswinds when cornering.")
            advice.append("🧤 Consider wearing glasses – wind-blown dust or debris can impair vision.")

        elif weather.wind_speed >= 5:
            advice.append("🍃 Breezy but manageable – good for cooling, minimal impact on riding.")
            advice.append("🔋 Slight headwind may increase effort on long straights or climbs.")
        
        return "\n".join(advice) if advice else "✅ Good riding conditions - enjoy your adventure!"