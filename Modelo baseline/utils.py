import pandas as pd
import os

script_dir = '.'


uri = os.path.join(script_dir, 'projects.csv')

try:

    dados = pd.read_csv(uri)

except FileNotFoundError:
    print("Erro: Arquivo não encontrado. Verifique o caminho do arquivo.")
except Exception as e:
    print(f"Ocorreu um erro ao ler o arquivo: {e}")