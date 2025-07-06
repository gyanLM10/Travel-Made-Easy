import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Agent.agentic_workflow import GraphBuilder
from starlette.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_travel_agent(query: QueryRequest):
    try:
        print(f"📥 Received query: {query.question}")

        # Initialize the agent workflow
        graph = GraphBuilder(model_provider="groq")
        react_app = graph()

        # Construct structured messages for the agent
        messages = {
            "messages": [
                {"role": "system", "content": "You are an expert travel planner. Provide detailed, multi-day travel itineraries with day-wise activities, hotel suggestions, transport tips, and food recommendations."},
                {"role": "user", "content": query.question}
            ]
        }

        # Run the agentic workflow
        output = react_app.invoke(messages)

        # Extract the final message
        if isinstance(output, dict) and "messages" in output:
            final_output = output["messages"][-1].content
        else:
            final_output = str(output)

        print("✅ Final output generated")
        return {"answer": final_output}

    except Exception as e:
        print("❌ Exception occurred:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})
