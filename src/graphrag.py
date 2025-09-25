# -*- coding: utf-8 -*-
"""
Módulo: faq_graph.py
Descrição: Construção e visualização de um grafo de FAQ em Python.
O grafo conecta perguntas e respostas em um modelo de Knowledge Graph.
"""

import os
import json
import networkx as nx
import matplotlib.pyplot as plt


def build_graph_from_faq(json_path: str) -> nx.Graph:
    """
    Constrói um grafo a partir de um arquivo JSON contendo perguntas e respostas.

    Parâmetros
    ----------
    json_path : str
        Caminho para o arquivo JSON contendo as perguntas e respostas.

    Retorno
    -------
    G : networkx.Graph
        Grafo contendo nós (perguntas e respostas) e arestas (relação entre eles).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    G = nx.Graph()

    for item in faq_data:
        pergunta = item["q"]
        resposta = item["a"]

        # Adiciona nós com tipos diferentes
        G.add_node(pergunta, type="pergunta")
        G.add_node(resposta, type="resposta")

        # Cria aresta entre pergunta e resposta
        G.add_edge(pergunta, resposta, relation="responde")

    return G


def plot_graph(G: nx.Graph, title: str = "FAQ Knowledge Graph") -> None:
    """
    Plota o grafo com estilo visual melhorado.

    Parâmetros
    ----------
    G : networkx.Graph
        Grafo a ser plotado.
    title : str, opcional
        Título do gráfico. Padrão é "FAQ Knowledge Graph".
    """
    pos = nx.spring_layout(G, seed=42, k=1.2)

    # Separação de nós por tipo
    perguntas = [n for n, d in G.nodes(data=True) if d["type"] == "pergunta"]
    respostas = [n for n, d in G.nodes(data=True) if d["type"] == "resposta"]

    plt.figure(figsize=(16, 12))

    # Perguntas em azul
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=perguntas,
        node_color="#1f77b4",
        node_size=2500,
        alpha=0.9,
        label="Perguntas (FAQ)"
    )

    # Respostas em verde
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=respostas,
        node_color="#2ca02c",
        node_size=1800,
        alpha=0.9,
        label="Respostas (Sistema)"
    )

    # Arestas
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.6)

    # Labels dos nós com quebra de linha
    labels = {
        n: "\n".join(n[i:i + 25] for i in range(0, len(n), 25))
        for n in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9,
                            font_color="white")

    # Título
    plt.title(title, fontsize=18, fontweight="bold")

    # Legenda clara e posicionada no canto inferior esquerdo
    plt.legend(
        scatterpoints=1,
        fontsize=12,
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="black"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Caminho para o arquivo JSON de FAQ
    json_path = r"C:\Users\rafae.RAFAEL_NOTEBOOK\Downloads\case_cd_welhome\py\faq.json"

    # Construir e plotar o grafo
    G = build_graph_from_faq(json_path)
    plot_graph(G)
