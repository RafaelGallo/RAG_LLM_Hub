# LLM_model.py
# Classe para encapsular o uso de LLMs (Google Gemini API)
# Responsável por geração de texto a partir de prompts

import os
import google.generativeai as genai


class LLMModel:
    """
    Classe de interface com o modelo Gemini.
    Utiliza a chave configurada em variáveis de ambiente (.env ou secrets.toml).
    """

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """
        Inicializa o cliente Gemini.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY não foi encontrada nas variáveis de ambiente.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        """
        Gera texto a partir de um prompt.
        """
        try:
            resp = self.model.generate_content(prompt)
            return resp.text.strip() if resp and resp.text else ""
        except Exception as e:
            return f"[Erro na geração de conteúdo: {e}]"
