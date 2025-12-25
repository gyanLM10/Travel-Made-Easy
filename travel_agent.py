import os
import sys
from dotenv import load_dotenv

# Ensure project root is on Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Agent.agentic_workflow import GraphBuilder

# Load environment variables
load_dotenv()

def get_travel_plan(question: str) -> str:
    try:
        print(f"\n📥 Received query: {question}")

        # Initialize the agent workflow with tools
        graph_builder = GraphBuilder(model_provider="groq")

        # ✅ Print all registered tools for debugging
        print("\n🔧 Registered Tools:")
        for tool in graph_builder.tools:
            print("✅", tool.name)

        # Build LangGraph flow
        graph = graph_builder()

        # Build system + user message sequence
        messages = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert travel planner with access to tools. "
                        "Use tools like `search_attractions`, `search_restaurants`, etc. "
                        "to answer queries accurately. Always prefer tool calls for external data."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        }

        # Run the agentic workflow
        output = graph.invoke(messages)

        # Extract final response
        if isinstance(output, dict) and "messages" in output:
            return output["messages"][-1].get("content", "⚠️ No content found in response.")
        else:
            return str(output)

    except Exception as e:
        print("❌ Exception occurred:", str(e))
        return f"Error: {str(e)}"
