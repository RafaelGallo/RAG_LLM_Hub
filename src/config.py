import os
from dotenv import load_dotenv
from huggingface_hub import login

# Carrega .env local (não afeta no Streamlit Cloud)
load_dotenv()

# GEMINI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Erro: variável GEMINI_API_KEY não encontrada.")

# HUGGING FACE
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)
if HF_TOKEN:
    try:
        login(HF_TOKEN)
    except Exception as e:
        print(f"Aviso: não foi possível logar no Hugging Face ({e}).")
else:
    print("Aviso: nenhum HUGGINGFACE_TOKEN encontrado. Modelos privados podem falhar.")
