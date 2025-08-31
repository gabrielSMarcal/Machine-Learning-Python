import pandas as pd
import os

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import numpy as np

SEED = 20

script_dir = '.'


uri = os.path.join(script_dir, 'projects.csv')

try:

    dados = pd.read_csv(uri)

except FileNotFoundError:
    print("Erro: Arquivo não encontrado. Verifique o caminho do arquivo.")
except Exception as e:
    print(f"Ocorreu um erro ao ler o arquivo: {e}")

dados['finalizados'] = dados['nao_finalizado'].map({1: 0, 0: 1})

dados_uteis = dados.query('horas_esperadas > 0')

x = dados_uteis[['horas_esperadas', 'preco']]
y = dados_uteis['finalizados']

raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.2, random_state=SEED, stratify=y)

scaler = StandardScaler()
scaler.fit(raw_treino_x)

treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

modelo = SVC(gamma='auto')
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

data_col1 = teste_x[: , 0]
data_col2 = teste_x[: , 1]

x_min = data_col1.min()
x_max = data_col1.max()
y_min = data_col2.min()
y_max = data_col2.max()

pixels = 100

eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)

xx, yy = np.meshgrid(eixo_x, eixo_y)

pontos = np.c_[xx.ravel(), yy.ravel()]

# Adicione esta linha para criar um DataFrame com nomes de colunas
pontos_df = pd.DataFrame(pontos, columns=['horas_esperadas', 'preco'])

Z = modelo.predict(pontos_df)
Z = Z.reshape(xx.shape)