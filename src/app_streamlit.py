"""
Interface Streamlit para o Welhome Assistant.
Integra RAG (FAISS + SentenceTransformers) e Gemini 2.0 Pro.
"""

import os
import streamlit as st
from rag_store import VectorStore
from config import GEMINI_API_KEY

# ============================
# Inicialização
# ============================
if "store" not in st.session_state:
    st.session_state.store = VectorStore(GEMINI_API_KEY)

# Carregar FAQ
faq_path = os.path.join("data", "faq.json")
if os.path.exists(faq_path):
    try:
        st.session_state.store.load_faq_from_json(faq_path)
        st.sidebar.success("✅ FAQ carregado com sucesso!")
    except Exception as e:
        st.sidebar.error(f"⚠️ Erro ao carregar FAQ: {e}")
else:
    st.sidebar.warning("⚠️ Nenhum FAQ encontrado em data/faq.json")

# Configuração visual
st.set_page_config(page_title="Welhome Assistant", layout="wide")
st.title("🏡 Welhome Assistant - RAG + Gemini 2.0 Pro")

# ============================
# Input do usuário
# ============================
query = st.text_input("Digite sua pergunta ou dúvida sobre imóveis:")

if st.button("🔍 Buscar resposta") and query:
    with st.spinner("Gerando resposta com Gemini 2.0 Pro..."):
        try:
            # Busca no FAISS
            similares = st.session_state.store.search(query, k=3)

            # Contexto para o LLM
            context = "\n".join(similares)
            prompt = f"""
            Você é um assistente da Welhome.
            Pergunta do usuário: {query}
            Contexto (FAQ + histórico): {context}
            Responda de forma clara, breve e útil.
            """

            resposta = st.session_state.store.llm.generate(prompt)

            # Exibir
            st.subheader("Resposta")
            st.write(resposta)

            # Histórico
            st.session_state.store.add_history(query, resposta)
            st.success("✅ Resposta salva no histórico!")

        except Exception as e:
            st.error(f"⚠️ Erro ao gerar resposta: {e}")

# ============================
# Histórico
# ============================
st.sidebar.header("📜 Histórico de consultas")

if st.session_state.store.get_history():
    for idx, h in enumerate(st.session_state.store.get_history(), 1):
        st.sidebar.markdown(f"**{idx}. {h['query']}**")
        st.sidebar.caption(h["resposta"])
else:
    st.sidebar.info("Nenhuma consulta realizada ainda.")
