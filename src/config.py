# config.py
import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if HF_TOKEN:
    login(HF_TOKEN)
else:
    print("Nenhum token do Hugging Face encontrado. Modelos podem falhar.")
