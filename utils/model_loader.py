import os
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables once (local .env)
load_dotenv()

# ── Streamlit Cloud secrets fallback ───────────────────────────────────────
# When running on Streamlit Cloud, .env is not deployed (it's gitignored).
# Instead, secrets are stored via the Streamlit Cloud dashboard and exposed
# via st.secrets. Inject them into os.environ so all downstream code works.
try:
    import streamlit as st
    _SECRET_KEYS = [
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
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
    pass  # Not running inside Streamlit, or secrets not configured yet
# ───────────────────────────────────────────────────────────────────────────


class ModelLoader(BaseModel):
    """
    Central LLM factory.
    Supported providers: 'gemini' (default), 'groq'
    """

    model_provider: Literal["gemini", "groq"] = "gemini"

    class Config:
        arbitrary_types_allowed = True

    def load_llm(self):
        print("🔁 Initializing LLM")
        print(f"🔧 Provider: {self.model_provider}")

        if self.model_provider == "gemini":
            return self._load_gemini()

        if self.model_provider == "groq":
            return self._load_groq()

        raise ValueError(f"Unsupported model provider: {self.model_provider}")

    # -------------------------
    # Providers
    # -------------------------

    def _load_gemini(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY not set")

        model_name = "gemini-2.0-flash"

        print(f"📡 Gemini model: {model_name}")

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.4,
            max_output_tokens=2000,
            timeout=60,
            google_api_key=api_key,
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
