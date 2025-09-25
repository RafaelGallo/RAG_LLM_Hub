# src/LLM_model.py
import os
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


class LLMModel:
    def __init__(self, api_key: str, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Classe centralizada para embeddings e geração de texto.
        - Sempre roda em CPU
        """
        # Configuração do Gemini
        genai.configure(api_key=api_key)
        self.gemini = genai.GenerativeModel("gemini-1.5-flash")

        # SentenceTransformer (sem device fixo)
        self.embedder = SentenceTransformer(embed_model)

    def embed(self, text: str):
        """Gera embedding sempre no CPU e retorna numpy array"""
        emb = self.embedder.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
            device="cpu"
        )[0]
        return emb

    def generate(self, prompt: str) -> str:
        """Gera resposta com Gemini"""
        resp = self.gemini.generate_content(prompt)
        return resp.text.strip()
