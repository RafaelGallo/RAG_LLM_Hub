from typing import Dict
import google.generativeai as genai


def init_gemini(api_key: str, model_name: str = "gemini-1.5-flash"):
    """
    Inicializa o modelo Gemini da API Google Generative AI.

    Args:
        api_key (str): Chave de API do Gemini carregada do .env
        model_name (str): Nome do modelo a ser utilizado (default: gemini-1.5-flash)

    Returns:
        genai.GenerativeModel: Instância configurada do modelo Gemini
    """
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def build_pitch(model, lead: Dict) -> str:
    """
    Gera um pitch personalizado para um lead com base nos dados fornecidos.

    Args:
        model: Instância do modelo Gemini
        lead (Dict): Dados do lead contendo nome, imóveis, localização e experiência

    Returns:
        str: Texto do pitch gerado pelo modelo
    """
    prompt = f"""
    Você é um assistente da Welhome.
    O lead forneceu:
    - Nome: {lead.get("nome")}
    - Imóveis: {lead.get("qtd_imoveis")}
    - Localização: {lead.get("localizacao")}
    - Experiência: {lead.get("experiencia")}

    Explique de forma clara e personalizada como a Welhome pode ajudar.
    Foque em: qualificação de leads, redução de tempo de venda e facilidade de uso do painel.
    """
    resp = model.generate_content(prompt)
    return resp.text.strip()


def summarize_for_sales(model, lead: Dict, pitch: str) -> Dict:
    """
    Gera um resumo estruturado e conciso do lead para uso pelo time de vendas.

    Args:
        model: Instância do modelo Gemini
        lead (Dict): Dados do lead
        pitch (str): Pitch gerado previamente

    Returns:
        Dict: Dicionário contendo o resumo estruturado e os dados originais do lead
    """
    prompt = f"""
    Gere um resumo estruturado e conciso (máx 6 linhas) para o vendedor.
    Dados do lead: {lead}
    Pitch gerado: {pitch}
    Formato esperado (campos fixos):
    - Nome
    - Qtd_Imoveis
    - Localizacao
    - Experiencia
    - Pontos_Chave (bullet points)
    - Proximos_Passos (bullet points)
    """
    resp = model.generate_content(prompt)
    texto = resp.text.strip()
    return {"resumo_texto": texto, **lead}
