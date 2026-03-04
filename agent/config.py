# src/config.py
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    llm_model: str = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash")
    gemini_model: str = os.getenv("GEMINI_MODEL", os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash"))

settings = Settings()
