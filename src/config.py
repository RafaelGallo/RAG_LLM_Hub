# config.py
import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Hugging Face
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    login(HF_TOKEN)
