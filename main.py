from sklearn.calibration import LinearSVC
import utils

modelo = LinearSVC()
modelo.fit(utils.dados, utils.classes)

animal_misterioso = [[1, 0, 1], [0, 0, 0], [1, 1, 0]]
resultado = modelo.predict(animal_misterioso)

print(f"Resultado: {resultado}")