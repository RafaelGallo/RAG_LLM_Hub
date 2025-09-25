# rag_store.py
# Módulo responsável pelo armazenamento vetorial (FAISS) e histórico de interações
# Utiliza SentenceTransformers para embeddings e integra com LLMModel (Gemini + HF)

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from LLM_model import LLMModel


class VectorStore:
    """
    Classe para gerenciar um banco de vetores com FAISS.
    Armazena perguntas/respostas do FAQ e histórico de leads.
    """

    def __init__(self):
        # Inicializa o modelo de embeddings
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Inicializa FAISS index (cosine similarity)
        self.index = faiss.IndexFlatIP(384)  # Dimensão do all-MiniLM-L6-v2 é 384
        self.data: List[Dict] = []

        # Inicializa o modelo LLM
        self.llm = LLMModel()

        # Histórico de leads
        self.history: Dict[str, str] = {}

    def embed(self, text: str) -> np.ndarray:
        """
        Gera o embedding de um texto usando SentenceTransformers.
        """
        emb = self.embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return emb

    def load_faq_from_json(self, json_path: str):
        """
        Carrega o FAQ a partir de um arquivo JSON.
        Estrutura esperada: [{"q": "...", "a": "..."}]
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Arquivo FAQ não encontrado: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            faq_data = json.load(f)

        for item in faq_data:
            q = item["q"]
            a = item["a"]
            emb = self.embed(q)
            self.index.add(emb)
            self.data.append({"question": q, "answer": a})

    def rag_answer(self, query: str, top_k: int = 2) -> Dict:
        """
        Recupera a resposta mais próxima do FAQ via FAISS.
        """
        q_emb = self.embed(query)
        D, I = self.index.search(q_emb, top_k)

        if len(I[0]) == 0:
            return {"question": None, "answer": "Não encontrei resposta no FAQ.", "candidates": []}

        candidates = []
        for idx in I[0]:
            if idx < len(self.data):
                candidates.append(self.data[idx])

        hit = candidates[0] if candidates else {"question": None, "answer": "Sem resposta"}
        return {"question": hit["question"], "answer": hit["answer"], "candidates": candidates}

    def add_history(self, lead_id: str, resumo: str):
        """
        Adiciona um resumo de lead ao histórico.
        """
        self.history[lead_id] = resumo

    def search_history(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Busca semântica no histórico de leads.
        """
        if not self.history:
            return []

        keys = list(self.history.keys())
        values = list(self.history.values())

        embs = self.embed(values)
        q_emb = self.embed(query)

        index_hist = faiss.IndexFlatIP(384)
        index_hist.add(embs)

        D, I = index_hist.search(q_emb, top_k)

        results = []
        for idx in I[0]:
            if idx < len(keys):
                results.append({"lead_id": keys[idx], "resumo": values[idx]})

        return results
