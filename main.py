from sklearn.calibration import LinearSVC
import utils

modelo = LinearSVC()
modelo.fit(utils.dados, utils.classes)

animal_misterioso = [[1, 0, 1], [0, 0, 0], [1, 1, 0]]
resultado = modelo.predict(animal_misterioso)

misterio1 = [1, 1, 1]
misterio2 = [1, 1, 0]
misterio3 = [0, 1, 1]

testes = [misterio1, misterio2, misterio3]

previsoes = modelo.predict(testes)

teste_classes = [0, 1, 1]

corretos = (previsoes == teste_classes).sum()
total = len(teste_classes)
taxa_de_acertos = (corretos / total) * 100

print(f"Acurácia: {taxa_de_acertos:.2f}%")