

from prompt_library.prompt import SYSTEM_PROMPT
from langchain_core.messages import HumanMessage


def weather_constraints(llm, weather_data):
    """
    Convert raw weather data into travel-relevant constraints.
    """
    return llm.invoke(
        f"""
You are a travel expert.

Analyze the weather data below and summarize it in 5 concise bullet points.
Focus ONLY on how weather impacts travel plans, clothing, and activities.
Avoid raw data repetition.

Weather data:
{weather_data}
"""
    ).content


def attraction_plan(llm, places_data, travel_style, days):
    """
    Create a day-wise attraction plan.
    """
    return llm.invoke(
        f"""
You are an expert travel planner.

Based on the attractions below and the travel style "{travel_style}",
create a {days}-day attraction plan.

Rules:
- Group attractions day-wise
- Balance busy and relaxed days
- Avoid listing too many places in one day
- Do NOT repeat raw text

Attractions data:
{places_data}
"""
    ).content


def stay_strategy(llm, hotels_data, budget):
    """
    Decide where and how to stay.
    """
    return llm.invoke(
        f"""
You are a travel accommodation expert.

Based on the hotel information below and a "{budget}" budget,
recommend:
- Best areas to stay
- Type of accommodation
- General price expectations

Avoid listing individual hotels unless necessary.

Hotel data:
{hotels_data}
"""
    ).content


def transport_strategy(llm, transport_data):
    """
    Decide local transport strategy.
    """
    return llm.invoke(
        f"""
You are a local transport expert.

Using the information below, design a transport strategy that:
- Minimizes travel fatigue
- Is cost-effective
- Is realistic for tourists

Transport data:
{transport_data}
"""
    ).content


def detailed_itinerary(
    llm,
    weather_summary,
    places_plan,
    stay_plan,
    transport_plan,
    days,
):
    """
    Final long-form itinerary.
    This is the ONLY step that uses SYSTEM_PROMPT.
    """

    messages = [
        SYSTEM_PROMPT,
        HumanMessage(
            content=f"""
Create a VERY DETAILED {days}-day travel plan using the information below.

IMPORTANT INSTRUCTIONS:
- Provide TWO itineraries:
  1) Popular tourist itinerary
  2) Off-beat / hidden gems itinerary
- Include hotels with approximate per-night cost
- Include restaurants with price range
- Include activities and transport details
- Include a clear cost breakdown and daily budget estimate
- Use clean Markdown formatting
- Do NOT invent missing data; state clearly if unavailable

Weather summary:
{weather_summary}

Attraction plan:
{places_plan}

Stay strategy:
{stay_plan}

Transport strategy:
{transport_plan}
"""
        ),
    ]

    return llm.invoke(messages).content
