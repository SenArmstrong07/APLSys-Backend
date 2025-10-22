from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1:free")

# single shared client instance
openrouter = OpenAI(
    base_url=OPENROUTER_BASE,
    api_key=OPENROUTER_API_KEY,
)