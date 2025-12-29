import os
from typing import List
from dotenv import load_dotenv
from langchain.tools import tool

from utils.place_info import GooglePlaceSearchTool, TavilyPlaceSearchTool


class PlaceSearchTool:
    def __init__(self):
        load_dotenv()

        self.google_api_key = os.environ.get("GPLACES_API_KEY")
        self.google_places_search = GooglePlaceSearchTool(self.google_api_key)
        self.tavily_search = TavilyPlaceSearchTool()

        self.place_search_tool_list = self._setup_tools()

    def _setup_tools(self) -> List:
        """Setup all tools for the place search tool"""

        @tool
        def search_attractions(place: str) -> str:
            """Search attractions of a place"""
            try:
                result = self.google_places_search.google_search_attractions(place)
                if result:
                    return f"Attractions in {place} (Google):\n{result}"
            except Exception as e:
                tavily_result = self.tavily_search.attractions(place)
                return (
                    f"Google failed due to {e}.\n"
                    f"Attractions in {place} (Tavily):\n{tavily_result}"
                )

        @tool
        def search_restaurants(place: str) -> str:
            """Search restaurants of a place"""
            try:
                result = self.google_places_search.google_search_restaurants(place)
                if result:
                    return f"Restaurants in {place} (Google):\n{result}"
            except Exception as e:
                tavily_result = self.tavily_search.restaurants(place)
                return (
                    f"Google failed due to {e}.\n"
                    f"Restaurants in {place} (Tavily):\n{tavily_result}"
                )

        @tool
        def search_activities(place: str) -> str:
            """Search activities of a place"""
            try:
                result = self.google_places_search.google_search_activity(place)
                if result:
                    return f"Activities in {place} (Google):\n{result}"
            except Exception as e:
                tavily_result = self.tavily_search.activities(place)
                return (
                    f"Google failed due to {e}.\n"
                    f"Activities in {place} (Tavily):\n{tavily_result}"
                )

        @tool
        def search_transportation(place: str) -> str:
            """Search transportation of a place"""
            try:
                result = self.google_places_search.google_search_transportation(place)
                if result:
                    return f"Transportation in {place} (Google):\n{result}"
            except Exception as e:
                tavily_result = self.tavily_search.transportation(place)
                return (
                    f"Google failed due to {e}.\n"
                    f"Transportation in {place} (Tavily):\n{tavily_result}"
                )

        return [
            search_attractions,
            search_restaurants,
            search_activities,
            search_transportation,
        ]
