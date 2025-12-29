# utils/planner/planner.py

from utils.planner.steps import (
    weather_constraints,
    attraction_plan,
    stay_strategy,
    transport_strategy,
    detailed_itinerary,
)


class TravelPlanner:
    """
    High-level planner that orchestrates multi-step LLM reasoning.
    """

    def __init__(self, llm):
        self.llm = llm

    def create_plan(
        self,
        destination: str,
        days: int,
        travel_style: str,
        budget: str,
        weather_data: dict,
        places_data: str,
        hotels_data: str,
        transport_data: str,
    ) -> str:
        """
        Generate a detailed travel itinerary.
        """

        # Step 1: Weather reasoning
        weather_summary = weather_constraints(
            self.llm,
            weather_data,
        )

        # Step 2: Attractions planning
        places_plan = attraction_plan(
            self.llm,
            places_data,
            travel_style,
            days,
        )

        # Step 3: Stay strategy
        stay_plan = stay_strategy(
            self.llm,
            hotels_data,
            budget,
        )

        # Step 4: Transport strategy
        transport_plan = transport_strategy(
            self.llm,
            transport_data,
        )

        # Step 5: Final itinerary synthesis
        final_plan = detailed_itinerary(
            self.llm,
            weather_summary,
            places_plan,
            stay_plan,
            transport_plan,
            days,
        )

        return final_plan
