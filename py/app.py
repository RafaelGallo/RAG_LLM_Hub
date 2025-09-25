import google.generativeai as genai
from py.config import GEMINI_API_KEY

# Configurar Gemini com chave do .env
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

resp = model.generate_content("Explique os benefícios da Welhome em 2 frases.")
print(resp.text)
