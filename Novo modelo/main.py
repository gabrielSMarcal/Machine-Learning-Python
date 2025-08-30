from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import utils

modelo = LinearSVC()
modelo.fit(utils.treino_x, utils.treino_y)

previsoes = modelo.predict(utils.teste_x)

acuracia = accuracy_score(utils.teste_y, previsoes) * 100

print(f"A acurácia do modelo é: {acuracia:.2f}%")
