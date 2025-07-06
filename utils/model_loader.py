import os
from dotenv import load_dotenv
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field
from utils.config_loader import load_config
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

class ConfigLoader:
    def __init__(self):
        print(f"✅ Loaded config...")
        self.config = load_config()

    def __getitem__(self, key):
        return self.config[key]

class ModelLoader(BaseModel):
    model_provider: Literal["groq", "openai"] = "groq"
    config: Optional[ConfigLoader] = Field(default=None, exclude=True)

    def model_post_init(self, __context: Any) -> None:
        self.config = ConfigLoader()

    class Config:
        arbitrary_types_allowed = True

    def load_llm(self):
        """
        Load and return the appropriate LLM based on the provider.
        """
        print("🔁 Loading LLM...")
        print(f"🔧 Model provider: {self.model_provider}")
        self.config = ConfigLoader()

        if self.model_provider == "groq":
            print("📡 Connecting to Groq API...")
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("❌ GROQ_API_KEY not found in environment.")

            model_name = "llama3-8b-8192"
            print(f"✅ Using Groq model: {model_name}")
            return ChatGroq(model=model_name)

        elif self.model_provider == "openai":
            print("📡 Connecting to OpenAI API...")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("❌ OPENAI_API_KEY not found in environment.")

            model_name = "gpt-4o"
            print(f"✅ Using OpenAI model: {model_name}")
            return ChatOpenAI(model=model_name)

        else:
            raise ValueError(f"❌ Unsupported model provider: {self.model_provider}")
