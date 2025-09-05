import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

d_novo = pd.read_csv('Novas_casas.csv', sep=';')
dados = pd.read_csv('Preços_de_casas.csv')

dados = dados.drop(columns='Id')
d_novo = d_novo.drop(columns='Casa')

# Correlação
corr = dados.corr()
corr_2 = d_novo.corr()

# # Quais fatores estão mais correlacionados?

# # Gerar uma máscara para o triângulo superior

# mascara = np.zeros_like(corr, dtype=bool)
# mascara[np.triu_indices_from(mascara)] = True

# # Configurar a figura do matplotlib
# f, ax = plt.subplots(figsize=(11, 9))

# # Gerar o mapa de calor (heatmap)

# cmap = sns.diverging_palette(220, 10, as_cmap=True)

# sns.heatmap(corr, mask=mascara, cmap=cmap, vmax=1, vmin=-1, center=0,
#             square=True, linewidths=.5, annot=True, cbar_kws={'shrink': .5})

# # Exibir o mapa de calor (heatmap)
# plt.show()
