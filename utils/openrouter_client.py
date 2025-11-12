from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1:free")

# Print API key presence for debugging
print(f"OpenRouter API key configured: {bool(OPENROUTER_API_KEY)}")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set in environment")


# single shared client instance
client = OpenAI(
    base_url=OPENROUTER_BASE,
    api_key=OPENROUTER_API_KEY,
)