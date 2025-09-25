# expand_faq.py
import json
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Carregar variáveis do arquivo .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("A chave GEMINI_API_KEY não foi encontrada no arquivo .env")

# Configuração da API do Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Carregar o arquivo faq.json existente no mesmo diretório
faq_path = os.path.join(os.path.dirname(__file__), "faq.json")
with open(faq_path, "r", encoding="utf-8") as f:
    faq_data = json.load(f)

# Lista que armazenará o dataset expandido
expanded_faq = []

# Inicialização do modelo do Gemini
model = genai.GenerativeModel("gemini-1.5-flash")

# Número máximo de perguntas no dataset final
MAX_Q = 1000
count = 0

for item in faq_data:
    pergunta = item["q"]
    resposta = item["a"]

    # Sempre manter a pergunta original
    expanded_faq.append({"q": pergunta, "a": resposta})
    count += 1

    # Gerar variações com o Gemini
    if count < MAX_Q:
        prompt = f"""
        Gere 10 variações diferentes da seguinte pergunta, mantendo o mesmo sentido:
        Pergunta: "{pergunta}"
        Responda apenas com a lista em português.
        """
        try:
            response = model.generate_content(prompt)
            variations = response.text.strip().split("\n")

            for v in variations:
                v = v.strip("-• ").strip()
                if v and count < MAX_Q:
                    expanded_faq.append({"q": v, "a": resposta})
                    count += 1

        except Exception as e:
            print(f"Erro ao gerar variações para '{pergunta}': {e}")

    if count >= MAX_Q:
        break

# Caminho de saída do arquivo expandido (na pasta data/)
output_path = os.path.join(os.path.dirname(__file__), "..", "data", "faq_expandido.json")
output_path = os.path.abspath(output_path)

# Salvar FAQ expandido em JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(expanded_faq, f, ensure_ascii=False, indent=2)

print(f"FAQ expandido salvo em {output_path} com {len(expanded_faq)} frases.")
