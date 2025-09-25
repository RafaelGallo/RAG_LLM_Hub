"""
Módulo responsável pelo armazenamento vetorial (FAISS)
e histórico de interações usando SentenceTransformers.
"""

import os
import json
import faiss
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from LLM_model import LLMModel


class VectorStore:
    """
    Classe que integra:
    - FAISS para busca vetorial
    - SentenceTransformers para embeddings
    - LLM Gemini 2.0 Pro para geração de respostas
    """

    def __init__(self, api_key: str, embed_model: str = "all-MiniLM-L6-v2") -> None:
        self.llm = LLMModel(api_key, model_name="gemini-2.0-pro")
        self.encoder = SentenceTransformer(embed_model)

        # FAISS
        self.index = None
        self.texts: List[str] = []

        # Histórico
        self.hist_items: List[Dict[str, str]] = []

    # ----------------------------
    # FAQ
    # ----------------------------
    def load_faq_from_json(self, json_path: str) -> None:
        """
        Carrega um FAQ em JSON e indexa no FAISS.
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Arquivo FAQ não encontrado: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            faq_data = json.load(f)

        # Suporta chaves "pergunta"/"resposta" ou "q"/"a"
        self.texts = [
            f"Q: {item.get('pergunta', item.get('q'))}\nA: {item.get('resposta', item.get('a'))}"
            for item in faq_data
        ]

        embeddings = self.encoder.encode(self.texts, convert_to_numpy=True)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    # ----------------------------
    # Busca
    # ----------------------------
    def search(self, query: str, k: int = 3) -> List[str]:
        """
        Busca no FAISS e retorna os textos mais similares.
        """
        if self.index is None:
            return ["❌ FAQ não foi carregado no índice."]

        emb = self.encoder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(emb, k)

        return [self.texts[i] for i in indices[0] if i < len(self.texts)]

    # ----------------------------
    # Histórico
    # ----------------------------
    def add_history(self, query: str, resposta: str) -> None:
        """
        Adiciona uma interação ao histórico.
        """
        self.hist_items.append({"query": query, "resposta": resposta})

    def get_history(self) -> List[Dict[str, str]]:
        """
        Retorna o histórico completo.
        """
        return self.hist_items
