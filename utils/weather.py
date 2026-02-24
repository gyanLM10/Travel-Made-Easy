import requests


class WeatherForecastTool:
    """
    Fetches weather data from OpenWeatherMap.
    Returns compact, structured data only.
    """

    def __init__(self, api_key: str):
        if not api_key:
            import warnings
            warnings.warn("OPENWEATHERMAP_API_KEY not provided – weather tools will return empty results.")

        self.api_key = api_key or ""
        self.base_url = "https://api.openweathermap.org/data/2.5"

    def get_current_weather(self, place: str) -> dict:
        """
        Get current weather of a place (compact).
        """
        try:
            response = requests.get(
                f"{self.base_url}/weather",
                params={
                    "q": place,
                    "appid": self.api_key,
                    "units": "metric",
                },
                timeout=10,
            )

            if response.status_code != 200:
                return {}

            data = response.json()

            return {
                "temperature": data.get("main", {}).get("temp"),
                "feels_like": data.get("main", {}).get("feels_like"),
                "weather": data.get("weather", [{}])[0].get("description"),
                "humidity": data.get("main", {}).get("humidity"),
            }

        except requests.RequestException:
            return {}

    def get_forecast_weather(self, place: str, days: int = 5) -> list:
        """
        Get short-term forecast (daily summary).
        """
        try:
            response = requests.get(
                f"{self.base_url}/forecast",
                params={
                    "q": place,
                    "appid": self.api_key,
                    "units": "metric",
                    "cnt": days * 8,  # approx daily data
                },
                timeout=10,
            )

            if response.status_code != 200:
                return []

            data = response.json()
            forecasts = []

            for item in data.get("list", []):
                forecasts.append({
                    "datetime": item.get("dt_txt"),
                    "temperature": item.get("main", {}).get("temp"),
                    "weather": item.get("weather", [{}])[0].get("description"),
                })

            return forecasts

        except requests.RequestException:
            return []
