import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import utils as u

SEED = 20

u.dados['finalizados'] = u.dados['nao_finalizado'].map({1: 0, 0: 1}) # Revertendo o valor para uma nova coluna

# print(u.dados.head()) # Criando a coluna 'finalizados'

# sns.scatterplot(x='horas_esperadas', y='preco', data=u.dados, hue='finalizados') # Plotando o gráfico de dispersão
# sns.relplot(x='horas_esperadas', y='preco', data=u.dados, hue='finalizados', col='finalizados') # Plotando o gráfico de dispersão

# plt.show()

dados_uteis = u.dados.query('horas_esperadas > 0')

x = dados_uteis[['horas_esperadas', 'preco']]
y = dados_uteis['finalizados']

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.2, random_state=SEED, stratify=y)

# Base teste_y é de 52,59%


print(f"Treinaremos com {len(treino_x)} elementos e testaremos com {len(teste_x)} elementos")

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)

print(treino_y.value_counts())
print(teste_y.value_counts())

previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100

print(f"A acurácia do modelo é: {acuracia:.2f}%")
