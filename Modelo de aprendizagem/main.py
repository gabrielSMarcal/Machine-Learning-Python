from sklearn.calibration import LinearSVC
from sklearn.metrics import accuracy_score
import utils

modelo = LinearSVC()
modelo.fit(utils.treino_x, utils.treino_y)

animal_misterioso = [[1, 0, 1], [0, 0, 0], [1, 1, 0]]
resultado = modelo.predict(animal_misterioso)

misterio1 = [1, 1, 1]
misterio2 = [1, 1, 0]
misterio3 = [0, 1, 1]

teste_x = [misterio1, misterio2, misterio3]
previsoes = modelo.predict(teste_x)

teste_y = [0, 1, 1]

# corretos = (previsoes == teste_y).sum()
# total = len(teste_y)
# taxa_de_acertos = corretos / total * 100
taxa_de_acertos = accuracy_score(teste_y, previsoes) * 100

print(f"Acurácia: {taxa_de_acertos:.2f}%")