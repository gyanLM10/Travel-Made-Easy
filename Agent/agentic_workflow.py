# agentic_workflow.py

import os
from dotenv import load_dotenv

# Load environment variables just once, ideally at program entry point
load_dotenv()

from utils.model_loader import ModelLoader
from prompt_library.prompt import SYSTEM_PROMPT  # Updated import path for prompt consistency
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.prebuilt import ToolNode, tools_condition

# Tool imports: Ensure these return .*_tool_list as per your PlaceSearchTool etc.
from tools.weather_info_tool import WeatherInfoTool
from tools.place_search_tool import PlaceSearchTool
from tools.expense_calculator_tool import CalculatorTool
from tools.currency_conversion_tool import CurrencyConverterTool


class GraphBuilder:
    def __init__(self, model_provider: str = "groq"):
        # 1. Load model
        self.model_loader = ModelLoader(model_provider=model_provider)
        self.llm = self.model_loader.load_llm()
        if self.llm is None:
            raise ValueError("LLM failed to initialize. Check GROQ_API_KEY or model provider.")

        # 2. Load tools from each tool class (each returns a list)
        self.weather_tools = WeatherInfoTool()
        self.place_search_tools = PlaceSearchTool()
        self.calculator_tools = CalculatorTool()
        self.currency_converter_tools = CurrencyConverterTool()

        self.tools = []
        self.tools.extend(self.weather_tools.weather_tool_list)
        self.tools.extend(self.place_search_tools.place_search_tool_list)
        self.tools.extend(self.calculator_tools.calculator_tool_list)
        self.tools.extend(self.currency_converter_tools.currency_converter_tool_list)

        # 3. Bind tools to LLM so agent can call them dynamically
        self.llm_with_tools = self.llm.bind_tools(tools=self.tools)

        # 4. Store system prompt for reuse
        self.system_prompt = SYSTEM_PROMPT

    def agent_function(self, state: MessagesState):
        """The main function for the AI agent in the graph."""
        user_messages = state["messages"]
        # Ensure the system prompt is always prepended for every query
        full_prompt = [self.system_prompt] + user_messages
        response = self.llm_with_tools.invoke(full_prompt)
        return {"messages": [response]}

    def build_graph(self):
        """Build and compile the conversation state graph."""
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("agent", self.agent_function)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))

        graph_builder.add_edge(START, "agent")
        graph_builder.add_conditional_edges("agent", tools_condition)
        graph_builder.add_edge("tools", "agent")
        graph_builder.add_edge("agent", END)

        self.graph = graph_builder.compile()
        return self.graph

    def __call__(self):
        return self.build_graph()


# Optional: standalone test run
if __name__ == "__main__":
    # Instantiate graph builder
    graph_builder = GraphBuilder()
    graph = graph_builder()

    # Simulate a user query
    user_input = "What are the best places to visit in Tokyo and their weather?"
    result = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
    print(result)
