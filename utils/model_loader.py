import os
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Load environment variables once (local .env)
load_dotenv()

# ── Streamlit Cloud secrets fallback ───────────────────────────────────────
try:
    import streamlit as st
    _SECRET_KEYS = [
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "GPLACES_API_KEY",
        "OPENWEATHERMAP_API_KEY",
        "EXCHANGE_RATE_API_KEY",
    ]
    for _key in _SECRET_KEYS:
        if _key not in os.environ:
            _val = st.secrets.get(_key)
            if _val:
                os.environ[_key] = _val
except Exception:
    pass
# ───────────────────────────────────────────────────────────────────────────


class ModelLoader(BaseModel):
    """
    Central LLM factory.
    Supported providers: 'openai' (default), 'groq'
    """

    model_provider: Literal["openai", "groq"] = "openai"

    class Config:
        arbitrary_types_allowed = True

    def load_llm(self):
        print("🔁 Initializing LLM")
        print(f"🔧 Provider: {self.model_provider}")

        if self.model_provider == "openai":
            return self._load_openai()

        if self.model_provider == "groq":
            return self._load_groq()

        raise ValueError(f"Unsupported model provider: {self.model_provider}")

    # -------------------------
    # Providers
    # -------------------------

    def _load_openai(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY not set")

        model_name = "gpt-4o-mini"

        print(f"📡 OpenAI model: {model_name}")

        return ChatOpenAI(
            model=model_name,
            temperature=0.4,
            max_tokens=2000,
            timeout=60
        )

    def _load_groq(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY not set")

        model_name = "llama-3.1-8b-instant"

        print(f"📡 Groq model: {model_name}")

        return ChatGroq(
            model=model_name,
            temperature=0.4,
            max_tokens=1500,
            timeout=60
        )
