import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from LLM_model import LLMModel


class VectorStore:
    """Armazena embeddings e textos em FAISS para RAG."""

    def __init__(self, api_key: str, embed_model: str = "all-MiniLM-L6-v2"):
        self.llm = LLMModel(api_key, embed_model)
        self.encoder = self.llm.encoder
        self.index = None
        self.texts = []
        self.hist_items = []  # histórico de interações

    def load_faq_from_json(self, file_path: str):
        """Carrega perguntas/respostas do FAQ e indexa em FAISS."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("O JSON do FAQ deve ser uma lista de objetos.")

            self.texts = [item["pergunta"] + " " + item.get("resposta", "")
                          for item in data]

            embeddings = [self.encoder.encode(t).astype("float32") for t in self.texts]
            embeddings = np.array(embeddings)

            # Cria índice FAISS
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings)

            print(f"✅ FAQ carregado e indexado com {len(self.texts)} entradas.")

        except Exception as e:
            raise RuntimeError(f"Erro ao carregar FAQ: {e}")

    def search(self, query: str, k: int = 3):
        """Busca similaridades no índice FAISS."""
        if self.index is None:
            raise ValueError("O índice FAISS ainda não foi criado.")

        query_emb = self.encoder.encode(query).astype("float32")
        query_emb = np.expand_dims(query_emb, axis=0)

        distances, indices = self.index.search(query_emb, k)
        results = [self.texts[i] for i in indices[0]]

        return results
