from datetime import datetime
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import graphviz

SEED = 20

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

# scaler = StandardScaler()
# scaler.fit(raw_treino_x)

# treino_x = scaler.transform(raw_treino_x)
# teste_x = scaler.transform(raw_teste_x)

modelo = DecisionTreeClassifier(max_depth=3)
modelo.fit(raw_treino_x, treino_y)
previsoes = modelo.predict(raw_teste_x)

accuracy = accuracy_score(teste_y, previsoes) * 100
print(f"A acurácia do modelo é: {accuracy:.2f}%")

estrutura = export_graphviz(modelo, filled=True, rounded=True, feature_names=x.columns, class_names=['não vendido', 'vendido'])
grafico = graphviz.Source(estrutura)

# Renderização do gráfico
grafico.render(filename='arvore_decisao', format='png', cleanup=True)
