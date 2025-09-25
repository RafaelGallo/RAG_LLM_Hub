import os
import google.generativeai as genai
from huggingface_hub import InferenceClient


class LLMModel:
    def __init__(self):
        # === Gemini API ===
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("⚠️ GEMINI_API_KEY não encontrada. Configure no Secrets.")

        genai.configure(api_key=gemini_key)

        # Usa o modelo mais rápido do Gemini
        try:
            self.gemini = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            print("Erro com gemini-1.5-flash:", e)
            self.gemini = genai.GenerativeModel("gemini-1.5-pro")

        # === Hugging Face ===
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            self.hf_client = InferenceClient(token=hf_token)
        else:
            self.hf_client = None
            print("⚠️ Nenhum HUGGINGFACE_TOKEN encontrado (Hugging Face pode falhar).")

    # -------- Gemini --------
    def generate(self, prompt: str) -> str:
        """Gera resposta com o Gemini"""
        try:
            resp = self.gemini.generate_content(prompt)
            return resp.text if hasattr(resp, "text") else str(resp)
        except Exception as e:
            return f"⚠️ Erro no Gemini: {e}"

    # -------- Hugging Face --------
    def generate_hf(self, prompt: str, model: str = "mistralai/Mistral-7B-Instruct-v0.2") -> str:
        """Gera resposta usando Hugging Face"""
        if not self.hf_client:
            return "⚠️ Hugging Face não configurado."
        try:
            resp = self.hf_client.text_generation(
                model=model,
                prompt=prompt,
                max_new_tokens=512,
                temperature=0.7
            )
            return resp
        except Exception as e:
            return f"⚠️ Erro no Hugging Face: {e}"
