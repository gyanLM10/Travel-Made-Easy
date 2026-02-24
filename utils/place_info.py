from langchain_tavily import TavilySearch
from langchain_google_community import GooglePlacesTool, GooglePlacesAPIWrapper


# -----------------------------
# Google Places
# -----------------------------
class GooglePlaceSearchTool:
    def __init__(self, api_key: str):
        self.places_wrapper = GooglePlacesAPIWrapper(gplaces_api_key=api_key)
        self.places_tool = GooglePlacesTool(api_wrapper=self.places_wrapper)

    def attractions(self, place: str) -> str:
        return self.places_tool.run(
            f"Top tourist attractions in {place}. Return short bullet points only."
        )

    def restaurants(self, place: str) -> str:
        return self.places_tool.run(
            f"Top restaurants and eateries in {place}. Return short bullet points only."
        )

    def activities(self, place: str) -> str:
        return self.places_tool.run(
            f"Popular activities in {place}. Return short bullet points only."
        )

    def transportation(self, place: str) -> str:
        return self.places_tool.run(
            f"Modes of local transportation in {place}. Return concise list."
        )

    # Aliases used by place_search_tool.py
    def google_search_attractions(self, place: str) -> str:
        return self.attractions(place)

    def google_search_restaurants(self, place: str) -> str:
        return self.restaurants(place)

    def google_search_activity(self, place: str) -> str:
        return self.activities(place)

    def google_search_transportation(self, place: str) -> str:
        return self.transportation(place)


# -----------------------------
# Tavily (Web Search)
# -----------------------------
class TavilyPlaceSearchTool:
    def __init__(self):
        self.tool = TavilySearch(topic="general", include_answer="basic")

    def _query(self, query: str) -> str:
        """
        Internal helper that always returns a SHORT string.
        Compatible with LangChain AIMessage output.
        """
        result = self.tool.invoke({"query": query})

        # LangChain returns AIMessage
        if hasattr(result, "content"):
            return result.content[:1500]

        # Fallback safety (should rarely happen)
        return str(result)[:1500]

    def attractions(self, place: str) -> str:
        return self._query(f"Top tourist attractions in {place}")

    def restaurants(self, place: str) -> str:
        return self._query(f"Best restaurants and local food in {place}")

    def activities(self, place: str) -> str:
        return self._query(f"Best activities and experiences in {place}")

    def transportation(self, place: str) -> str:
        return self._query(f"Local transportation options in {place}")


