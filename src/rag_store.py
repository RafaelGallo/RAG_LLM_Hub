"""
Módulo: rag_store.py
Descrição: Implementação de um Vector Store baseado em FAISS para integração com LLM.
O Vector Store é responsável por armazenar embeddings, realizar buscas semânticas
e suportar operações de RAG (Retrieval-Augmented Generation).
"""

import faiss
import numpy as np
import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from LLM_model import LLMModel


class VectorStore:
    """
    Classe responsável por gerenciar embeddings com FAISS, incluindo:
    - FAQ em memória
    - Histórico de leads
    - Suporte a busca semântica
    """

    def __init__(self, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Inicializa o Vector Store com modelo de embeddings e LLM.
        """
        self.embed_model = embed_model
        self.encoder = SentenceTransformer(embed_model)
        self.llm = LLMModel()  # LLMModel lê a chave do ambiente automaticamente

        # Estruturas internas
        self.index = None
        self.faq: List[Dict] = []
        self.history: List[Dict] = []

    def load_faq_from_json(self, json_path: str) -> None:
        """
        Carrega perguntas e respostas de um arquivo JSON para a memória
        e cria o índice FAISS com embeddings.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            self.faq = json.load(f)

        questions = [item["q"] for item in self.faq]
        embeddings = self.encoder.encode(questions, convert_to_numpy=True)

        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)

    def rag_answer(self, question: str, top_k: int = 2) -> Dict:
        """
        Recupera a resposta mais próxima do FAQ com base na similaridade semântica.
        """
        if not self.index:
            raise ValueError("O índice FAISS não foi inicializado. Carregue o FAQ primeiro.")

        q_emb = self.encoder.encode([question], convert_to_numpy=True)
        distances, indices = self.index.search(q_emb, top_k)

        candidates = []
        for idx in indices[0]:
            if idx < len(self.faq):
                candidates.append(self.faq[idx])

        best = candidates[0] if candidates else {"q": "", "a": "Nenhuma resposta encontrada."}

        return {
            "question": best["q"],
            "answer": best["a"],
            "candidates": candidates,
        }

    def add_history(self, lead_id: str, resumo: str) -> None:
        """
        Armazena um resumo no histórico com embedding associado.
        """
        emb = self.encoder.encode([resumo], convert_to_numpy=True)
        self.history.append({"lead_id": lead_id, "resumo": resumo, "embedding": emb})

    def search_history(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Realiza busca semântica no histórico de leads.
        """
        if not self.history:
            return []

        query_emb = self.encoder.encode([query], convert_to_numpy=True)
        sims = []

        for h in self.history:
            sim = float(np.dot(query_emb, h["embedding"].T) / (np.linalg.norm(query_emb) * np.linalg.norm(h["embedding"])))
            sims.append((sim, h))

        sims.sort(key=lambda x: x[0], reverse=True)
        results = [h for _, h in sims[:top_k]]

        return results
