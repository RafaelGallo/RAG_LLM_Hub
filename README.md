# Welhome Assistant 

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/) [![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)](https://faiss.ai/) [![SentenceTransformers](https://img.shields.io/badge/Sentence--Transformers-Embeddings-green)](https://www.sbert.net/) [![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co/transformers/) [![Gemini](https://img.shields.io/badge/Google-Gemini%20API-brightgreen?logo=google)](https://ai.google.dev/) [![Matplotlib](https://img.shields.io/badge/Matplotlib-Charts-blue)](https://matplotlib.org/) [![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Analysis-purple)](https://networkx.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


O **Welhome Assistant** é um projeto de demonstração que combina **LLMs (Gemini)**, **RAG (FAISS)** e visualização de grafos (**GraphRAG**) para oferecer um assistente inteligente de vendas.  
Ele foi desenvolvido como um **case de entrevista técnica**, demonstrando habilidades de **engenharia de prompts, integração de LLMs, RAG, e visualização de conhecimento**.

## Funcionalidades

- **Chatbot (LLM - Gemini)**  
  Gera um pitch personalizado para cada lead e cria um resumo estruturado para auxiliar vendedores.  

- **RAG (FAQ Expandido)**  
  Responde perguntas frequentes dos leads a partir de um FAQ expandido (até **1000 variações** de perguntas).  

- **Histórico de Leads**  
  Armazena interações em um banco vetorial (FAISS) e permite exportar para **CSV**.  

- **GraphRAG**  
  Gera um **grafo de conhecimento** que conecta perguntas e respostas do FAQ, facilitando a visualização.  

## Estrutura do Projeto

```

├── src/
│   ├── app_streamlit.py      # Interface principal (Streamlit)
│   ├── app_cli.py            # Interface de linha de comando
│   ├── rag_store.py          # Vetorstore (FAISS)
│   ├── LLM_model.py          # Wrapper para Gemini + embeddings
│   └── expand_faq.py         # Script para expandir FAQ
│
├── data/
│   ├── faq.json              # FAQ original
│   └── faq_expandido.json    # FAQ expandido (até 1000 frases)
│
├── py/
│   └── config.py             # Configurações (.env e tokens)
│
├── img/
│   ├── 001.png               # Tela inicial (Chatbot)
│   ├── 002.png               # Resumo gerado
│   ├── 003.png               # RAG (FAQ)
│   ├── 004.png               # Histórico de leads
│
├── output/
│   └── graph_LLM.png         # Grafo de conhecimento (FAQ)
│
├── requirements.txt
├── main.py                   # Launcher oficial (Streamlit)
└── README.md

````

## Instalação

### 1. Clonar o repositório
```bash
git clone https://github.com/RafaelGallo/RAG_LLM_Hub.git
cd RAG_LLM_Hub
````

### 2. Criar ambiente virtual

```bash
python -m venv agenteLLM2
agenteLLM2\Scripts\activate      # Windows
source agenteLLM2/bin/activate   # Linux/Mac
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente

Crie um arquivo **`.env`** na raiz com as chaves:

```
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
HUGGINGFACE_TOKEN=YOUR_HF_TOKEN
```

## Modelos Utilizados

Este projeto combina **LLMs (Large Language Models)** com **Transformers** para RAG:

* **Google Gemini 2.0 Pro**
  Utilizado como modelo de geração de linguagem natural (LLM principal).
  [Documentação Gemini](https://ai.google.dev/)

* **SentenceTransformers - all-MiniLM-L6-v2**
  Usado para geração de embeddings e indexação vetorial com FAISS.
  [SentenceTransformers](https://www.sbert.net/)

* **Hugging Face Transformers**
  Biblioteca base para carregamento dos modelos e integração com embeddings.
  [Transformers](https://huggingface.co/transformers/)

## Principais Dependências

* `google-generativeai` – API oficial do Gemini
* `sentence-transformers==2.5.1`
* `transformers==4.41.2`
* `faiss-cpu==1.12.0`
* `torch>=2.4.1`
* `streamlit==1.38.0`
* `networkx==3.3`, `matplotlib` (para análise de grafos e plotagens)

## Como rodar

```bash
# Criar e ativar ambiente virtual
conda create -n agenteLLM2 python=3.10 -y
conda activate agenteLLM2

# Instalar dependências
pip install -r requirements.txt

# Rodar a aplicação Streamlit
streamlit run src/app_streamlit.py
```

### Interface Web (Streamlit)

```bash
python main.py
```

A aplicação abrirá em `http://localhost:8501`.

### Interface CLI (linha de comando)

```bash
python src/app_cli.py
```

### Expandir FAQ até 1000 frases

```bash
python src/expand_faq.py
```

### Gerar Grafo (GraphRAG)

```bash
python py/graph_faq.py
```

## 📸 Demonstrações

### 1. Chatbot (Pitch + Resumo)

![Chatbot](img/001.png)

Essas imagens mostram a interface em Streamlit.
A primeira tela destaca o formulário de entrada de leads com campos de nome, quantidade de imóveis, localização e experiência.

### 2. Resumo Estruturado

![Resumo](img/002.png)

A segunda mostra a geração automática de pitch e resumo estruturado, que pode ser usado diretamente pela equipe de vendas.
O layout é limpo, com abas para Chatbot, RAG e Histórico, facilitando o uso.

### 3. RAG (FAQ)

![RAG](img/003.png)

Demonstra a funcionalidade de busca no FAQ expandido (RAG).
O usuário faz uma pergunta, e o sistema encontra a resposta mais relevante com base nas embeddings.
Além disso, a resposta é naturalizada pelo modelo Gemini, tornando-a clara e amigável para o lead.

### 4. Histórico de Leads

![Histórico](img/004.png) 

Mostra a aba de histórico com exportação em CSV.
Cada interação com leads é salva, permitindo acompanhamento e análise de marketing posterior.
Esse recurso conecta a parte de gestão de vendas com insights automáticos.

### 5. Knowledge Graph (FAQ)

![GraphRAG](output/graph_LLM.png)

É a representação visual do grafo de conhecimento construído a partir do FAQ.
Os nós azuis representam perguntas e os nós verdes representam respostas, conectados por arestas "responde".
Esse gráfico ajuda a entender como as informações estão relacionadas e pode ser expandido para análises mais complexas de GraphRAG.

## Tecnologias Utilizadas

* **Python 3.10+**
* [Streamlit](https://streamlit.io/)
* [FAISS](https://faiss.ai/)
* [Sentence Transformers](https://www.sbert.net/)
* [Google Generative AI (Gemini)](https://ai.google.dev/)
* [Hugging Face Hub](https://huggingface.co/)
* [NetworkX](https://networkx.org/) + [Matplotlib](https://matplotlib.org/)

## 📜 Licença

MIT License © 2025 - [Rafael Gallo](https://github.com/RafaelGallo)
