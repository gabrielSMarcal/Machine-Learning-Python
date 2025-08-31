import pandas as pd
import os

script_dir = '.'

uri_tracking = os.path.join(script_dir, 'tracking.csv')
uri_precos = os.path.join(script_dir, 'precos.csv')
uri_projetos = os.path.join(script_dir, 'projects.csv')

try:
    dados_tracking = pd.read_csv(uri_tracking)
    dados_precos = pd.read_csv(uri_precos)
    dados_projetos = pd.read_csv(uri_projetos)

    y = dados_tracking['comprou']
    x = dados_tracking.drop(columns=['comprou'])
    
    treino_x = x[:75]
    treino_y = y[:75]

    teste_x = x[75:]
    teste_y = y[75:]

except FileNotFoundError:
    print("Erro: Arquivo não encontrado. Verifique o caminho do arquivo.")
except Exception as e:
    print(f"Ocorreu um erro ao ler o arquivo: {e}")