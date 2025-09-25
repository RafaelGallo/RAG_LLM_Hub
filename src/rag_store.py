# rag_store.py
"""
Módulo para gerenciamento do Vector Store (FAISS) com embeddings e histórico.
Integra SentenceTransformers para embeddings e Gemini para geração de respostas.
"""

import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from config import GEMINI_API_KEY, HF_TOKEN
from LLM_model import LLMModel


class VectorStore:
    """
    Classe para gerenciar embeddings, histórico e busca semântica usando FAISS.
    """

    def __init__(
        self,
        api_key: str,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Inicializa o Vector Store.

        Args:
            api_key (str): Chave da API do Gemini.
            embed_model (str): Nome do modelo de embeddings.
        """
        self.llm = LLMModel(api_key)

        # Fallback automático para modelo público caso não haja HF_TOKEN
        if HF_TOKEN:
            self.encoder = SentenceTransformer(embed_model)
        else:
            print(
                "Aviso: Nenhum token Hugging Face detectado. "
                "Usando modelo público mais leve."
            )
            self.encoder = SentenceTransformer(
                "sentence-transformers/paraphrase-MiniLM-L3-v2"
            )

        self.index = None
        self.history: Dict[str, str] = {}
        self.faq: List[Dict[str, str]] = []

    # -------------------------------------------------------------------------
    # FAQ
    # -------------------------------------------------------------------------
    def load_faq_from_json(self, path: str) -> None:
        """
        Carrega FAQ de um arquivo JSON.

        Args:
            path (str): Caminho do arquivo JSON.
        """
        with open(path, "r", encoding="utf-8") as f:
            self.faq = json.load(f)

        if not self.faq:
            raise ValueError("O arquivo FAQ está vazio.")

        self._build_faq_index()

    def _build_faq_index(self) -> None:
        """
        Constrói o índice FAISS a partir das perguntas do FAQ.
        """
        questions = [item["q"] for item in self.faq]
        embeddings = self.encoder.encode(questions, convert_to_numpy=True)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def rag_answer(self, question: str, top_k: int = 2) -> Dict[str, Any]:
        """
        Busca a resposta mais relevante no FAQ.

        Args:
            question (str): Pergunta de entrada.
            top_k (int): Número de candidatos retornados.

        Returns:
            dict: Contendo a pergunta, resposta e candidatos.
        """
        if not self.index:
            raise ValueError("Índice FAISS não foi inicializado.")

        q_emb = self.encoder.encode([question], convert_to_numpy=True)
        distances, indices = self.index.search(q_emb, top_k)

        candidates = []
        for idx in indices[0]:
            item = self.faq[idx]
            candidates.append({"question": item["q"], "answer": item["a"]})

        best = candidates[0]
        return {"question": best["question"], "answer": best["answer"], "candidates": candidates}

    # -------------------------------------------------------------------------
    # Histórico
    # -------------------------------------------------------------------------
    def add_history(self, lead_id: str, resumo: str) -> None:
        """
        Adiciona resumo ao histórico.

        Args:
            lead_id (str): Identificador do lead.
            resumo (str): Texto do resumo.
        """
        self.history[lead_id] = resumo

    def search_history(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """
        Busca semântica no histórico de leads.

        Args:
            query (str): Texto de busca.
            top_k (int): Número de resultados.

        Returns:
            list: Resultados contendo ID e resumo.
        """
        if not self.history:
            return []

        ids = list(self.history.keys())
        textos = list(self.history.values())
        embeddings = self.encoder.encode(textos, convert_to_numpy=True)

        dim = embeddings.shape[1]
        hist_index = faiss.IndexFlatL2(dim)
        hist_index.add(embeddings)

        q_emb = self.encoder.encode([query], convert_to_numpy=True)
        distances, indices = hist_index.search(q_emb, top_k)

        results = []
        for idx in indices[0]:
            lead_id = ids[idx]
            results.append({"lead_id": lead_id, "resumo": self.history[lead_id]})

        return results
