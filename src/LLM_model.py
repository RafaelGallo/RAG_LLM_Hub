"""
LLM_model.py
Módulo responsável por interagir com o modelo Gemini da Google
e gerar respostas/resumos a partir de prompts.
"""

import google.generativeai as genai
from sentence_transformers import SentenceTransformer


class LLMModel:
    """Classe para interação com LLM (Gemini)."""

    def __init__(self, api_key: str, embed_model: str):
        # Configuração da API Gemini
        genai.configure(api_key=api_key)

        # ✅ Corrigido: modelo precisa do prefixo "models/"
        self.gemini = genai.GenerativeModel("models/gemini-1.5-flash")

        # Encoder de embeddings (Hugging Face)
        self.encoder = SentenceTransformer(embed_model)

    def generate(self, prompt: str) -> str:
        """
        Gera uma resposta/resumo usando o modelo Gemini.
        """
        try:
            response = self.gemini.generate_content(prompt)
            return response.text if response and response.text else "⚠️ Resposta vazia."
        except Exception as e:
            return f"[Erro na geração de conteúdo: {str(e)}]"

    def embed(self, text: str):
        """
        Gera embeddings usando SentenceTransformer.
        """
        return self.encoder.encode(text).tolist()
