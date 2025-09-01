from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

SEED = 20

# DummyClassifier

dados = pd.read_csv('precos.csv')

# Adicionando novas colunas
dados['km_por_ano'] = dados['milhas_por_ano'] * 1.60934
dados['idade'] = datetime.now().year - dados['ano_do_modelo']

# Tratamento de colunas, removendo aquelas que não interessa
dados.drop(columns=['milhas_por_ano', 'ano_do_modelo'], axis=1, inplace=True)

x = dados[['preco', 'idade', 'km_por_ano']]
y = dados['vendido']

raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.2, random_state=SEED, stratify=y)

print(f'Treinaremos com {len(raw_treino_x)} elementos e testaremos com {len(raw_teste_x)} elementos.')


classificador = DummyClassifier(strategy='stratified')
classificador.fit(raw_treino_x, treino_y)
previsoes = classificador.predict(raw_teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100

print(f'A acurácia do modelo é de {acuracia:.2f}%')