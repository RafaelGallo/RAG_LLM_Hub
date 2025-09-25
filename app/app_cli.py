# app_cli.py
# CLI para o Case Welhome — LLM Gemini + RAG (FAISS)
# Requisitos: pip install -r requirements.txt
# Configuração: criar um arquivo .env com GEMINI_API_KEY (ver .env.example)

from py.config import GEMINI_API_KEY
from chatbot import init_gemini, build_pitch, summarize_for_sales
from rag_store import VectorStore
import json


def main():
    # Inicializa o modelo Gemini e o vetor semântico
    model = init_gemini(GEMINI_API_KEY, "gemini-1.5-flash")
    store = VectorStore(GEMINI_API_KEY)
    store.load_faq_from_json("data/faq.json")

    print("=== Chatbot Welhome (CLI) ===")

    # Entrada de dados do lead
    lead_id = input("ID do lead (ex: lead_001): ").strip() or "lead_001"
    lead = {
        "nome": input("Nome: ").strip(),
        "qtd_imoveis": input("Quantos imóveis deseja anunciar?: ").strip(),
        "localizacao": input("Localização (cidade/região): ").strip(),
        "experiencia": input("Já usou outras plataformas? ").strip(),
    }

    # Geração do pitch e resumo estruturado
    pitch = build_pitch(model, lead)
    resumo = summarize_for_sales(model, lead, pitch)

    resumo_texto = f"""Resumo do lead {lead_id}
{json.dumps(lead, ensure_ascii=False)}
Pitch:
{pitch}

Resumo estruturado:
{resumo.get('resumo_texto')}
"""

    # Armazenamento no histórico vetorial
    store.add_history(lead_id, resumo_texto)

    print("\n--- Pitch Personalizado ---\n")
    print(pitch)
    print("\n--- Resumo Estruturado (para vendedor) ---\n")
    print(resumo.get("resumo_texto"))

    # Recuperação aumentada (RAG) no FAQ
    print("\n=== RAG – Pergunte algo do FAQ (ENTER para pular) ===")
    q = input("Pergunta: ").strip()
    if q:
        hit = store.rag_answer(q, top_k=2)

        natural_prompt = f"""
        Você é o assistente da Welhome.
        Pergunta do lead: {q}
        Resposta do FAQ: {hit['answer']}
        Reescreva de forma clara, objetiva e amigável (2-4 linhas).
        """
        natural_resp = model.generate_content(natural_prompt).text.strip()

        print("\n[RAG] Pergunta FAQ mais próxima:", hit["question"])
        print("[RAG] Resposta naturalizada:", natural_resp)

    # Busca semântica no histórico de leads
    print("\n=== Busca no histórico (ENTER para pular) ===")
    hq = input("O que deseja buscar no histórico? ").strip()
    if hq:
        results = store.search_history(hq, top_k=3)
        print(f"\n{len(results)} resultado(s):")
        for r in results:
            print(f"- {r['lead_id']}")
        if results:
            print("\n--- Trecho do 1º resultado ---\n")
            print(results[0]["resumo"])


if __name__ == "__main__":
    main()
