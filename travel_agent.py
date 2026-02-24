import os
import sys
from dotenv import load_dotenv

# Ensure project root is on Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Agent.agentic_workflow import GraphBuilder

# Load environment variables
load_dotenv()


def get_travel_plan(question: str) -> str:
    """
    Runs the agentic travel planning workflow and returns a final string response.
    This function GUARANTEES that no AIMessage object is leaked outside.
    """
    try:
        print(f"\n📥 Received query: {question}")

        # Initialize the agent workflow
        graph_builder = GraphBuilder(model_provider="groq")

        # Debug: list registered tools
        print("\n🔧 Registered Tools:")
        for tool in graph_builder.tools:
            print("✅", tool.name)

        # Build LangGraph
        graph = graph_builder()

        # Input messages (LangGraph-compatible)
        messages = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert travel planner with access to tools. "
                        "Use tools like `search_attractions`, `search_restaurants`, "
                        "`search_activities`, and `search_transportation` whenever needed. "
                        "Always prefer tool calls for external and factual information."
                    ),
                },
                {
                    "role": "user",
                    "content": question,
                },
            ]
        }

        # Run the agentic workflow
        output = graph.invoke(messages)

        # ---- SAFE EXTRACTION (AIMessage FIX) ----
        if isinstance(output, dict) and "messages" in output:
            last_message = output["messages"][-1]

            # LangChain AIMessage → use .content
            if hasattr(last_message, "content"):
                return last_message.content

            # Fallback safety
            return str(last_message)

        # If output is not message-based
        return str(output)

    except Exception as e:
        print("❌ Exception occurred:", str(e))
        return f"Error: {str(e)}"


def get_travel_plan_with_validation(question: str) -> dict:
    """
    Runs the travel planner, then passes the result through the critic LLM
    for hallucination scoring.

    Returns:
        {
            "plan":       str   — the travel plan text,
            "validation": dict  — confidence score & uncertain claims
        }
    """
    from utils.response_validator import validate

    plan = get_travel_plan(question)
    validation = validate(question=question, plan=plan)

    return {"plan": plan, "validation": validation}
