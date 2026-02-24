# 🌍 Travel Made Easy

An AI-powered travel planning assistant built with **LangGraph**, **LangChain**, and **Streamlit**. Given a natural-language request like *"5-day trip to Paris with budget breakdown"*, it autonomously calls real-time tools — weather APIs, Google Places, currency converters — and assembles a full itinerary in Markdown. Every response is also scored by a second LLM critic to flag potential hallucinations.

🔗 **Live App:** [gyanlm10-travel-made-easy.streamlit.app](https://gyanlm10-travel-made-easy.streamlit.app)

---

## Architecture

```
User Query (text or audio)
        │
        ▼
   app.py  (Streamlit UI)
        │
        ▼
travel_agent.py → get_travel_plan_with_validation()
        │                          │
        ▼                          ▼
 GraphBuilder                ResponseValidator
 (LangGraph)                 (Critic LLM @ temp=0)
        │                          │
   ┌────┴────┐                confidence score
   │  agent  │◄──────────── system prompt
   └────┬────┘                (prompt_library/)
        │ tool calls
        ▼
   ┌─────────────────────────┐
   │   ToolNode (LangGraph)  │
   ├─────────────────────────┤
   │  search_attractions     │ ← Google Places + Tavily fallback
   │  search_restaurants     │ ← Google Places + Tavily fallback
   │  search_activities      │ ← Google Places + Tavily fallback
   │  search_transportation  │ ← Google Places + Tavily fallback
   │  get_current_weather    │ ← OpenWeatherMap
   │  get_weather_forecast   │ ← OpenWeatherMap (5-day)
   │  convert_currency       │ ← ExchangeRate-API v6
   │  estimate_total_hotel_cost     │ ← local arithmetic
   │  calculate_total_expense       │ ← local arithmetic
   │  calculate_daily_expense_budget│ ← local arithmetic
   └─────────────────────────┘
```

The graph runs a **ReAct loop**: the agent decides which tools to call, LangGraph executes them via `ToolNode`, results are fed back, and the agent keeps reasoning until it produces a final answer. Tool calls are automatic — the agent decides on its own what data it needs.

---

## Features

| Feature | Detail |
|---|---|
| **Dual Itinerary** | Always produces two plans: mainstream tourist route + off-beat local alternative |
| **Live Weather** | Current conditions + 5-day forecast via OpenWeatherMap |
| **Place Search** | Attractions, restaurants, activities, transport via Google Places (Tavily fallback on failure) |
| **Currency Conversion** | Real-time rates via ExchangeRate-API v6 |
| **Budget Calculator** | Hotel cost, total trip cost, daily budget — all computed by the agent automatically |
| **Hallucination Control** | Second LLM call at `temperature=0` scores responses 0–100, flags unverifiable claims |
| **Voice Input** | Audio upload (WAV/MP3/M4A) transcribed via OpenAI Whisper (optional) |

---

## Project Structure

```
Travel-Made-Easy/
├── app.py                    # Streamlit UI + render_plan() with trustworthiness panel
├── travel_agent.py           # get_travel_plan() + get_travel_plan_with_validation()
├── runtime.txt               # Pins Python 3.11 on Streamlit Cloud
├── requirements.txt          # pip dependencies
│
├── Agent/
│   └── agentic_workflow.py   # GraphBuilder: builds & compiles the LangGraph
│
├── prompt_library/
│   └── prompt.py             # SYSTEM_PROMPT (SystemMessage with full instructions)
│
├── tools/                    # LangChain @tool wrappers (each returns a *_tool_list)
│   ├── place_search_tool.py  # search_attractions/restaurants/activities/transportation
│   ├── weather_info_tool.py  # get_current_weather, get_weather_forecast
│   ├── currency_conversion_tool.py  # convert_currency
│   └── expense_calculator_tool.py   # estimate_total_hotel_cost, calculate_total_expense,
│                                    # calculate_daily_expense_budget
│
├── utils/                    # Backend services called by tools
│   ├── model_loader.py       # LLM factory: ChatGroq / ChatOpenAI + st.secrets fallback
│   ├── place_info.py         # GooglePlaceSearchTool + TavilyPlaceSearchTool
│   ├── weather.py            # WeatherForecastTool → OpenWeatherMap REST calls
│   ├── currency_converter.py # CurrencyConverter → ExchangeRate-API v6
│   ├── calculator_util.py    # Calculator (multiply, sum, daily budget)
│   ├── response_validator.py # ResponseValidator: critic LLM, returns confidence score
│   └── speech_to_text.py     # transcribe_audio() using openai-whisper (optional)
│
├── config/                   # Config loading utilities
├── logger/                   # Logging setup
└── exception/                # Custom exception classes
```

---

## LLM Configuration

Configured in `utils/model_loader.py` via `ModelLoader`:

| Provider | Model | Temp | Max Tokens |
|---|---|---|---|
| **OpenAI** (default) | `gpt-4o-mini` | 0.4 | 2000 |
| **Groq** (fallback) | `llama-3.1-8b-instant` | 0.4 | 1500 |
| **Critic** (hallucination) | `gpt-4o-mini` | **0** | 512 |

Switch provider in `travel_agent.py`:
```python
graph_builder = GraphBuilder(model_provider="openai")   # or "groq"
```

---

## Tool Design Pattern

Every tool class follows the same pattern so `GraphBuilder` can compose them uniformly:

```python
class WeatherInfoTool:
    def __init__(self):
        self.weather_service = WeatherForecastTool(api_key)
        self.weather_tool_list = self._setup_tools()   # list of @tool functions

    def _setup_tools(self) -> List:
        @tool
        def get_current_weather(city: str) -> str: ...
        return [get_current_weather, get_weather_forecast]
```

`GraphBuilder` aggregates all `*_tool_list` arrays and binds them to the LLM with `llm.bind_tools(tools)`.

### Place Search Fallback

`search_attractions` (and the other 3 place tools) try **Google Places** first; on failure they transparently fall back to **Tavily** search:

```python
try:
    result = google_places.google_search_attractions(place)
except Exception as e:
    result = tavily_search.attractions(place)   # automatic fallback
```

---

## Hallucination Control

After every plan is generated, `utils/response_validator.py` runs a second LLM call:

```python
# temperature=0 → deterministic critic evaluation
critic_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)
```

Returns structured JSON:
```json
{
  "confidence_score": 72,
  "trustworthy": true,
  "uncertain_claims": ["Eiffel Tower ticket €28 — may vary by season"],
  "verified_by_tools": ["weather", "restaurants"],
  "summary": "Plan is reasonable but verify prices before booking."
}
```

The app displays a colour-coded badge below every plan:

| Score | Badge |
|---|---|
| 80–100 | 🟢 High Confidence |
| 50–79 | 🟡 Verify Before Booking |
| 0–49 | 🔴 Low Confidence |

If the critic fails for any reason, it returns a safe default and the plan is still shown normally.

---

## API Keys Required

| Key | Where to Get | Used By |
|---|---|---|
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) | Planner (gpt-4o-mini) + Critic |
| `GPLACES_API_KEY` | [Google Cloud Console](https://console.cloud.google.com) → Places API | Place search |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) | Place search fallback |
| `OPENWEATHERMAP_API_KEY` | [openweathermap.org](https://openweathermap.org/api) | Weather tools |
| `EXCHANGE_RATE_API_KEY` | [exchangerate-api.com](https://exchangerate-api.com) | Currency conversion |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | Optional: Groq fallback provider |

---

## Local Setup

```bash
# 1. Clone
git clone https://github.com/gyanLM10/Travel-Made-Easy.git
cd Travel-Made-Easy

# 2. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add API keys
cp .env.name .env
# Edit .env and fill in all API keys

# 5. Run
streamlit run app.py
```

---

## Streamlit Cloud Deployment

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select `app.py`
3. Under **Settings → Secrets**, add all API keys in TOML format:

```toml
OPENAI_API_KEY = "sk-..."
GPLACES_API_KEY = "AIza..."
TAVILY_API_KEY = "tvly-..."
OPENWEATHERMAP_API_KEY = "..."
EXCHANGE_RATE_API_KEY = "..."
```

`model_loader.py` automatically reads `st.secrets` and injects keys into `os.environ` so all downstream code works without modification.

---

## Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Agent Framework | LangGraph (`StateGraph`, `MessagesState`, `ToolNode`) |
| LLM | OpenAI (`gpt-4o-mini`) via `langchain-openai` |
| Place Search | `langchain-google-community[places]` + Tavily |
| Weather | OpenWeatherMap REST API |
| Currency | ExchangeRate-API v6 |
| Voice | OpenAI Whisper (optional) |
| Python | 3.11 |

---

## License

MIT
