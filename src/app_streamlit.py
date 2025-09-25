"""
App Streamlit para RAG com Gemini 2.0 Pro + FAISS + SentenceTransformers.
"""

import os
import streamlit as st
from rag_store import VectorStore


# ======================
# Configurações iniciais
# ======================
st.set_page_config(
    page_title="RAG LLM Hub",
    page_icon="🤖",
    layout="wide",
)

# Carregar variáveis de ambiente do Streamlit Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)

if not GEMINI_API_KEY:
    st.error("❌ Chave GEMINI_API_KEY não encontrada no Secrets!")
    st.stop()

# Inicializar o VectorStore (apenas uma vez por sessão)
if "store" not in st.session_state:
    st.session_state.store = VectorStore(GEMINI_API_KEY)

# ======================
# Carregar FAQ
# ======================
faq_path = os.path.join("src", "data", "faq.json")

try:
    st.session_state.store.load_faq_from_json(faq_path)
    st.sidebar.success("✅ FAQ carregado com sucesso!")
except Exception as e:
    st.sidebar.error(f"⚠️ Erro ao carregar FAQ: {e}")

# ======================
# Interface
# ======================
st.title("🔎 RAG LLM Hub - Gemini 2.0 Pro")

st.markdown(
    """
    Este aplicativo utiliza **Google Gemini 2.0 Pro** como modelo LLM,  
    embeddings com **SentenceTransformers** e busca vetorial com **FAISS**.
    """
)

# Caixa de entrada do usuário
query = st.text_input("Digite sua pergunta:", placeholder="Exemplo: Como funciona o Welhome Assistant?")

if st.button("Pesquisar", type="primary"):
    if query.strip():
        with st.spinner("🔎 Buscando respostas no FAQ..."):
            # Buscar no FAISS
            results = st.session_state.store.search(query, k=3)

            # Gerar resumo com Gemini
            context = "\n".join(results)
            prompt = f"Baseando-se no FAQ abaixo, responda a pergunta:\n\n{context}\n\nPergunta: {query}"

            try:
                resumo = st.session_state.store.llm.generate(prompt)
                st.success("✅ Resposta gerada com sucesso!")

                st.subheader("📄 Resposta:")
                st.write(resumo)

                # Histórico
                st.session_state.store.hist_items.append({"query": query, "resposta": resumo})

            except Exception as e:
                st.error(f"⚠️ Erro ao gerar resposta: {e}")
    else:
        st.warning("Por favor, digite uma pergunta.")

# ======================
# Histórico
# ======================
st.subheader("📜 Histórico de interações")

if hasattr(st.session_state.store, "hist_items") and st.session_state.store.hist_items:
    for idx, item in enumerate(st.session_state.store.hist_items, 1):
        st.markdown(f"**{idx}. Pergunta:** {item['query']}")
        st.markdown(f"➡️ **Resposta:** {item['resposta']}")
        st.markdown("---")
else:
    st.info("Nenhuma interação registrada ainda.")
