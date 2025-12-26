# 🌍 Travel-Made-Easy — AI-Powered Trip Planner

Travel-Made-Easy is an **AI agent-based travel assistant** that helps users generate personalized travel plans. It uses **Groq's LLaMA3-8B**, **LangChain agents**, and real-time APIs for currency, weather, and local activities — all wrapped with a **FastAPI backend** and **Streamlit UI**.

---

## ✨ Features

- 🧠 **Agentic Workflow using LangChain**
  - Thoughtful multi-step reasoning via structured graphs
  - Supports memory and tool calling

- 🔧 **Integrated Real-Time Tools**
  - 🌦️ **Weather Tool** – Get weather forecasts for any location
  - 💱 **Currency Converter Tool** – Convert currencies using latest exchange rates
  - 💸 **Expense Calculator Tool** – Breakdown travel costs
  - 📍 **Place Search Tool** – Fetch local attractions and places to visit
  - ➗ **Arithmetic Tool** – Handles numerical reasoning (e.g., cost splitting)

- 🔌 **LLM Integration**
  - Powered by **Groq's LLaMA3-8B**
  - Support for **OpenAI** as fallback (optional)

- ⚙️ **FastAPI Backend**
  - Central query handler (`/query`)
  - Handles tool invocation and message flow

- 🎨 **Streamlit Frontend**
  - Simple, interactive interface for input/output
  - Can be deployed publicly (e.g., Streamlit Community Cloud)

---

## 🧠 How It Works

1. The user types a travel-related query.
2. Request hits FastAPI endpoint: `/query`
3. LangChain agent is created using:
   - LLaMA3 (via Groq)
   - Connected tools (weather, currency, etc.)
4. Agent processes the query, invokes tools if needed.
5. Final travel plan (structured by days or insights) is returned.
6. Streamlit UI renders it beautifully.

---

## 🛠️ Tech Stack

| Layer        | Tool/Library                |
|--------------|-----------------------------|
| 💬 LLM       | Groq ('llama-3.1-8b-instant')     |
| 🧠 Agent     | LangChain (`RunnableGraph`) |
| 🧰 Tools     | Custom Python-based tools   |
| ⚙️ Backend   | FastAPI                     |
| 🎨 Frontend  | Streamlit                   |
| 🔐 Secrets   | dotenv (`.env`)             |

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/travel-made-easy.git
cd travel-made-easy

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8040
streamlit run app.py

🧪 Example Prompts
"Plan a 3-day trip to Paris with cultural activities and budget in INR."

"How much will a 5-day New York trip cost if I convert it to Euros?"

"What are fun things to do in Tokyo this weekend?"

📂 Project Structure
bash
Copy
Edit
├── main.py                 # FastAPI backend
├── app.py                  # Streamlit frontend
├── Agent/
│   └── agentic_workflow.py # LangChain agent logic
├── tools/
│   ├── weather_info_tool.py
│   ├── currency_conversion_tool.py
│   ├── expense_calculator_tool.py
│   ├── place_search_tool.py
│   └── arithmetic_op.py
├── utils/
│   ├── config_loader.py
│   ├── model_loader.py
│   └── currency_converter.py
├── prompt_library/         
├── .env    

