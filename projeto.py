import pandas as pd
import os

# Obtém o diretório do script atual
script_dir = os.path.dirname(os.path.abspath(__file__))

# Constrói o caminho completo para os arquivos CSV
uri_tracking = os.path.join(script_dir, 'tracking.csv')
uri_projetos = os.path.join(script_dir, 'projects.csv')

# Tenta ler o arquivo CSV
try:
    dados_tracking = pd.read_csv(uri_tracking)
    dados_projetos = pd.read_csv(uri_projetos)
    print("Dados lidos com sucesso!")
    print(dados_tracking.head())  # Imprime as primeiras linhas para verificar
    print(dados_projetos.head())
except FileNotFoundError:
    print("Erro: Arquivo não encontrado. Verifique o caminho do arquivo.")
except Exception as e:
    print(f"Ocorreu um erro ao ler o arquivo: {e}")