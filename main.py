# main.py
import os
import subprocess
import sys


def main():
    """
    Launcher para rodar o aplicativo Streamlit do Welhome Assistant.
    """
    print("Iniciando Welhome Assistant...")

    # Caminho para o arquivo principal do Streamlit
    app_file = os.path.join("src", "app_streamlit.py")
    if not os.path.exists(app_file):
        print(f"Arquivo {app_file} não encontrado.")
        sys.exit(1)

    # Executa o Streamlit apontando para app_streamlit.py
    try:
        subprocess.run(["streamlit", "run", app_file], check=True)
    except Exception as e:
        print("Erro ao iniciar o Streamlit:", e)


if __name__ == "__main__":
    print("Iniciando Welhome Assistant via Streamlit...\n")

    script_path = os.path.join("src", "app_streamlit.py")
    subprocess.run(["streamlit", "run", script_path])
