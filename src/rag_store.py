"""
rag_store.py
Módulo que implementa a camada de armazenamento vetorial
para gerenciar histórico e embeddings.
"""

from typing import List, Dict
from LLM_model import LLMModel


class VectorStore:
    """Armazena embeddings, resumos e histórico de leads."""

    def __init__(self, api_key: str, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Inicializa o modelo LLM + encoder
        self.llm = LLMModel(api_key, embed_model)

        # Histórico de resumos gerados
        self.hist_items: List[Dict[str, str]] = []   # ✅ Garantido que sempre existe

    def add_history(self, lead_id: str, resumo: str) -> None:
        """
        Adiciona item ao histórico.
        """
        self.hist_items.append({"lead_id": lead_id, "resumo": resumo})

    def get_history(self) -> List[Dict[str, str]]:
        """
        Retorna todo o histórico armazenado.
        """
        return self.hist_items

    def summarize_and_store(self, lead_id: str, texto: str) -> str:
        """
        Gera um resumo com o Gemini e armazena no histórico.
        """
        resumo = self.llm.generate(texto)
        self.add_history(lead_id, resumo)
        return resumo
