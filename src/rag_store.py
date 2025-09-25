# src/rag_store.py
import os
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple
from LLM_model import LLMModel   # ✅ import direto, sem "src."


class VectorStore:
    def __init__(self, api_key: str,
                 embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 base_dir="base"):
        """
        Vector DB para FAQ e histórico de leads.
        - Salva embeddings (FAISS) + metadados (JSON) em 'base/'.
        - Sempre roda em CPU (via LLMModel).
        """
        self.llm = LLMModel(api_key, embed_model)

        # Armazenamento FAQ
        self.faq_index = None
        self.faq_items: List[Tuple[str, str]] = []

        # Armazenamento histórico
        self.hist_index = None
        self.hist_items: List[Dict] = []

        # Pasta base/
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.hist_index_path = os.path.join(self.base_dir, "history.index")
        self.hist_meta_path = os.path.join(self.base_dir, "history.json")

        # Se já existe histórico salvo, carrega
        self._load_history_from_disk()

    # =================== Embeddings ===================
    def _embed(self, text: str) -> np.ndarray:
        return np.array(self.llm.embed(text), dtype="float32")

    # =================== FAQ ===================
    def load_faq_from_json(self, path: str):
        """Carrega base de FAQ de um JSON e cria índice FAISS."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.faq_items = [(d["q"], d["a"]) for d in data]
        embs = np.vstack([self._embed(q) for q, _ in self.faq_items])

        self.faq_index = faiss.IndexFlatL2(embs.shape[1])
        self.faq_index.add(embs)

    def rag_answer(self, query: str, top_k: int = 1) -> Dict[str, str]:
        """Busca no FAQ e reescreve resposta em tom natural com Gemini."""
        if self.faq_index is None:
            raise RuntimeError("FAQ index não inicializado. Chame load_faq_from_json primeiro.")

        q_emb = self._embed(query).reshape(1, -1)
        D, I = self.faq_index.search(q_emb, top_k)
        hits = [self.faq_items[i] for i in I[0]]
        best_q, best_a = hits[0]

        # Reformulação
        prompt = f"""
        Pergunta do lead: {query}
        Resposta do FAQ: {best_a}
        Reescreva em tom natural, amigável e claro (2-4 linhas),
        como se fosse o assistente da Welhome respondendo.
        """
        answer = self.llm.generate(prompt)

        return {
            "question": best_q,
            "answer": answer,
            "candidates": [{"q": q, "a": a} for q, a in hits],
        }

    # =================== Histórico ===================
    def add_history(self, lead_id: str, resumo_texto: str):
        emb = self._embed(resumo_texto).reshape(1, -1)
        if self.hist_index is None:
            self.hist_index = faiss.IndexFlatL2(emb.shape[1])
        self.hist_index.add(emb)
        self.hist_items.append({"lead_id": lead_id, "resumo": resumo_texto})
        self._save_history_to_disk()

    def search_history(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        if self.hist_index is None or len(self.hist_items) == 0:
            return []
        q_emb = self._embed(query).reshape(1, -1)
        D, I = self.hist_index.search(q_emb, top_k)
        return [self.hist_items[i] for i in I[0] if 0 <= i < len(self.hist_items)]

    # =================== Persistência ===================
    def _save_history_to_disk(self):
        if self.hist_index:
            faiss.write_index(self.hist_index, self.hist_index_path)
            with open(self.hist_meta_path, "w", encoding="utf-8") as f:
                json.dump(self.hist_items, f, ensure_ascii=False, indent=2)

    def _load_history_from_disk(self):
        if os.path.exists(self.hist_index_path) and os.path.exists(self.hist_meta_path):
            self.hist_index = faiss.read_index(self.hist_index_path)
            with open(self.hist_meta_path, "r", encoding="utf-8") as f:
                self.hist_items = json.load(f)
