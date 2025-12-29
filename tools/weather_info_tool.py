import os
from typing import List
from dotenv import load_dotenv
from langchain.tools import tool

from utils.weather import WeatherForecastTool


class WeatherInfoTool:
    def __init__(self):
        load_dotenv()
        self.api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
        self.weather_service = WeatherForecastTool(self.api_key)
        self.weather_tool_list = self._setup_tools()

    def _setup_tools(self) -> List:
        """Setup all tools for the weather forecast tool"""

        @tool
        def get_current_weather(city: str) -> str:
            """Get current weather for a city"""
            data = self.weather_service.get_current_weather(city)

            if not data:
                return f"Could not fetch current weather for {city}"

            return (
                f"Current weather in {city}: "
                f"{data.get('temperature')}°C, "
                f"{data.get('weather')}, "
                f"humidity {data.get('humidity')}%"
            )

        @tool
        def get_weather_forecast(city: str) -> str:
            """Get short weather forecast summary for a city"""
            forecast = self.weather_service.get_forecast_weather(city, days=5)

            if not forecast:
                return f"Could not fetch forecast for {city}"

            summary_lines = []
            for item in forecast[:5]:
                summary_lines.append(
                    f"{item.get('datetime')}: "
                    f"{item.get('temperature')}°C, "
                    f"{item.get('weather')}"
                )

            return (
                    f"Short weather forecast for {city}:\n"
                    + "\n".join(summary_lines)
            )

        return [get_current_weather, get_weather_forecast]
