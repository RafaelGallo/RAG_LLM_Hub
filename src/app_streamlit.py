# src/app_streamlit.py
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

from rag_store import VectorStore   # ✅ sem "src."

# ================================
# Configuração inicial
# ================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if "store" not in st.session_state:
    st.session_state.store = VectorStore(GEMINI_API_KEY)
    # Carregar FAQ (se existir)
    faq_path = "data/faq.json"
    if os.path.exists(faq_path):
        st.session_state.store.load_faq_from_json(faq_path)

st.set_page_config(page_title="Welhome Assistant", layout="wide")

st.title("🏠 Welhome Assistant (Case de Entrevista)")

# ================================
# Tabs
# ================================
tab1, tab2, tab3 = st.tabs(["🤖 Chatbot", "📚 RAG (FAQ)", "📂 Histórico"])

# ================================
# TAB 1 - Chatbot
# ================================
with tab1:
    st.header("🤖 Chatbot")

    col1, col2 = st.columns(2)
    with col1:
        lead_id = st.text_input("ID do lead", "lead_001")
        nome = st.text_input("Nome")
        qtde = st.text_input("Qtde. de imóveis")
    with col2:
        localizacao = st.text_input("Localização")
        experiencia = st.text_input("Experiência")

    if st.button("Gerar Pitch + Resumo e Armazenar"):
        if nome and localizacao and experiencia and qtde:
            # Prompt para Gemini
            prompt = f"""
            Lead: {nome}, localizado em {localizacao}.
            Experiência anterior: {experiencia}.
            Possui {qtde} imóveis.
            Gere um pitch curto (3-5 linhas) explicando como a Welhome pode ajudá-lo.
            """
            resumo_texto = st.session_state.store.llm.generate(prompt)

            st.subheader("Resumo gerado")
            st.write(resumo_texto)

            # Salvar no histórico vetorial
            st.session_state.store.add_history(lead_id, resumo_texto)
            st.success("Resumo armazenado com sucesso!")
        else:
            st.error("Por favor, preencha todos os campos.")

# ================================
# TAB 2 - RAG (FAQ)
# ================================
with tab2:
    st.header("📚 RAG (FAQ)")

    user_q = st.text_input("Digite sua pergunta sobre a Welhome")
    if st.button("Buscar no FAQ"):
        if user_q:
            result = st.session_state.store.rag_answer(user_q)
            st.subheader("Resposta")
            st.write(result["answer"])

            with st.expander("Respostas candidatas"):
                st.json(result["candidates"])
        else:
            st.error("Digite uma pergunta.")

# ================================
# TAB 3 - Histórico
# ================================
with tab3:
    st.header("📂 Histórico de Leads")

    hist = st.session_state.store.hist_items
    if len(hist) > 0:
        df = pd.DataFrame(hist)
        st.dataframe(df, use_container_width=True)

        # Exportar para CSV
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Baixar histórico em CSV",
            csv,
            "historico_leads.csv",
            "text/csv",
            key="download-csv"
        )
    else:
        st.info("Nenhum lead armazenado ainda.")
