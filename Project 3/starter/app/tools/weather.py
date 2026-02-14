# app/tools/weather.py
from semantic_kernel.functions import kernel_function
import requests
import json
from app.utils.logger import setup_logger

logger = setup_logger("weather_tool")

class WeatherTools:
    @kernel_function(name="get_weather", description="Get 7-day weather forecast for a given city.")
    def get_weather(self, city: str) -> str:
        """
        Gets the 7-day weather forecast for a given city and returns a simple summary string.

        Args:
            city: Name of the city to get weather for

        Returns:
            JSON string with weather data including temperature, conditions, and recommendation
        """
        logger.info(f"Weather tool called with city={city}")

        try:
            # Step 1: Geocoding - get lat/lon from city name
            geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
            geocode_response = requests.get(geocode_url, timeout=10)
            geocode_response.raise_for_status()
            geocode_data = geocode_response.json()

            if "results" not in geocode_data or len(geocode_data["results"]) == 0:
                logger.warning(f"City not found: {city}")
                return json.dumps({
                    "error": f"City '{city}' not found",
                    "temperature_c": None,
                    "conditions": "unknown",
                    "recommendation": "Unable to get weather data"
                })

            lat = geocode_data["results"][0]["latitude"]
            lon = geocode_data["results"][0]["longitude"]
            city_name = geocode_data["results"][0].get("name", city)
            country = geocode_data["results"][0].get("country", "")
            logger.info(f"Geocoded {city} to lat={lat}, lon={lon}")

            # Step 2: Weather API call using coordinates
            weather_url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&daily=weathercode,temperature_2m_max,temperature_2m_min"
                f"&forecast_days=7&timezone=UTC"
            )
            weather_response = requests.get(weather_url, timeout=10)
            weather_response.raise_for_status()
            weather_data = weather_response.json()

            # Step 3: Process weather data
            daily = weather_data.get("daily", {})
            temps_max = daily.get("temperature_2m_max", [])
            temps_min = daily.get("temperature_2m_min", [])
            weather_codes = daily.get("weathercode", [])

            if not temps_max or not temps_min:
                return json.dumps({
                    "error": "No temperature data available",
                    "temperature_c": None,
                    "conditions": "unknown",
                    "recommendation": "Unable to get weather data"
                })

            # Calculate average temperature
            avg_max = sum(temps_max) / len(temps_max)
            avg_min = sum(temps_min) / len(temps_min)
            avg_temp = (avg_max + avg_min) / 2

            # Determine dominant weather condition from codes
            # Weather codes: ≤1 = Sunny, ≤3 = Cloudy, >50 = Rainy, else Mixed
            def interpret_code(code):
                if code <= 1:
                    return "sunny"
                elif code <= 3:
                    return "cloudy"
                elif code > 50:
                    return "rainy"
                else:
                    return "mixed"

            conditions_list = [interpret_code(c) for c in weather_codes]
            # Find most common condition
            condition_counts = {}
            for c in conditions_list:
                condition_counts[c] = condition_counts.get(c, 0) + 1
            dominant_condition = max(condition_counts, key=condition_counts.get)

            # Generate recommendation based on conditions
            if dominant_condition == "sunny":
                recommendation = "Great weather! Pack sunscreen and light clothing."
            elif dominant_condition == "cloudy":
                recommendation = "Expect overcast skies. Bring layers for varying temperatures."
            elif dominant_condition == "rainy":
                recommendation = "Rain expected. Pack an umbrella and waterproof jacket."
            else:
                recommendation = "Mixed conditions expected. Pack for various weather types."

            # Add temperature-specific advice
            if avg_temp < 10:
                recommendation += " Cold temperatures - bring warm clothing."
            elif avg_temp > 30:
                recommendation += " Hot weather - stay hydrated and seek shade."

            result = {
                "city": city_name,
                "country": country,
                "temperature_c": round(avg_temp, 1),
                "temperature_max_c": round(avg_max, 1),
                "temperature_min_c": round(avg_min, 1),
                "conditions": dominant_condition,
                "recommendation": recommendation,
                "forecast_days": len(temps_max)
            }

            logger.info(f"Weather result: {avg_temp:.1f}°C, {dominant_condition}")
            return json.dumps(result)

        except requests.RequestException as e:
            logger.error(f"Weather API request failed: {e}")
            return json.dumps({
                "error": f"Weather API request failed: {str(e)}",
                "temperature_c": None,
                "conditions": "unknown",
                "recommendation": "Unable to fetch weather data"
            })
        except Exception as e:
            logger.error(f"Weather tool error: {e}")
            return json.dumps({
                "error": f"Weather tool error: {str(e)}",
                "temperature_c": None,
                "conditions": "unknown",
                "recommendation": "An error occurred"
            })
