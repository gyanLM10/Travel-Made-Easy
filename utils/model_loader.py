import os
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Load environment variables once
load_dotenv()


class ModelLoader(BaseModel):
    """
    Central LLM factory.
    This class must ONLY create and return LLM instances.
    """

    model_provider: Literal["groq", "openai"] = "groq"

    class Config:
        arbitrary_types_allowed = True

    def load_llm(self):
        print("🔁 Initializing LLM")
        print(f"🔧 Provider: {self.model_provider}")

        if self.model_provider == "groq":
            return self._load_groq()

        if self.model_provider == "openai":
            return self._load_openai()

        raise ValueError(f"Unsupported model provider: {self.model_provider}")

    # -------------------------
    # Providers
    # -------------------------

    def _load_groq(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY not set")

        model_name = "llama-3.1-8b-instant"

        print(f"📡 Groq model: {model_name}")

        return ChatGroq(
            model=model_name,
            temperature=0.4,
            max_tokens=1500,     # 🔒 token safety
            timeout=60
        )

    def _load_openai(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY not set")

        model_name = "gpt-4-turbo"

        print(f"📡 OpenAI model: {model_name}")

        return ChatOpenAI(
            model=model_name,
            temperature=0.4,
            max_tokens=2000,     # 🔒 larger final synthesis
            timeout=60
        )
