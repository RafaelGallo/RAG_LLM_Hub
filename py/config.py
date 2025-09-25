# config.py
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Chave da API do Gemini (obrigatória)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY não encontrada. Defina no arquivo .env.")

# Token do Hugging Face (opcional, usado para autenticação de modelos privados)
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    # Login automático no Hugging Face Hub
    login(HF_TOKEN)
else:
    print("Aviso: Nenhum HUGGINGFACE_TOKEN encontrado no .env. "
          "Modelos privados no Hugging Face podem não funcionar.")
