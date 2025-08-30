from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import utils

SEED = 42655

utils.treino_x, utils.teste_x, utils.treino_y, utils.teste_y = train_test_split(utils.x, utils.y, random_state=SEED, stratify=utils.y)

print(f"Treinaremos com {len(utils.treino_x)} elementos e testaremos com {len(utils.teste_x)} elementos")

modelo = LinearSVC()
modelo.fit(utils.treino_x, utils.treino_y)

print(utils.treino_y.value_counts())
print(utils.teste_y.value_counts())

previsoes = modelo.predict(utils.teste_x)

acuracia = accuracy_score(utils.teste_y, previsoes) * 100

print(f"A acurácia do modelo é: {acuracia:.2f}%")