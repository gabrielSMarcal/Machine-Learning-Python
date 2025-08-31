from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import utils as u

'''PREVISÃO COM SVC'''

# print(f'Treinaremos com {len(u.treino_x)} elementos e testaremos com {len(u.teste_x)} elementos.')

# acuracia = accuracy_score(u.teste_y, u.previsoes) * 100
# print(f'A acurácia do modelo é: {acuracia:.2f}%')

'''VISUALIZAÇÃO DOS RESULTADOS EM GAMMA AUTO'''

plt.contourf(u.xx, u.yy, u.Z)
plt.scatter(u.data_col1, u.data_col2, c=u.teste_y, s=1)
plt.show()

