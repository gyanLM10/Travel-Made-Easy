import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Agent.agentic_workflow import GraphBuilder

load_dotenv()

def get_travel_plan(question: str) -> str:
    try:
        print(f"📥 Received query: {question}")

        # Initialize the agent workflow
        graph = GraphBuilder(model_provider="groq")
        react_app = graph()

        # Construct structured messages for the agent
        messages = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert travel planner. Provide detailed, multi-day travel itineraries with day-wise activities, hotel suggestions, transport tips, and food recommendations."
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        }

        # Run the agentic workflow
        output = react_app.invoke(messages)

        # Extract final message
        if isinstance(output, dict) and "messages" in output:
            return output["messages"][-1].content
        else:
            return str(output)

    except Exception as e:
        print("❌ Exception occurred:", str(e))
        return f"Error: {str(e)}"
