import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


import pandas as pd
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

import utils as u

# Como é a relação entre área construida e o preço do imovel?

# plt.scatter(u.dados['area_primeiro_andar'], u.dados['preco_de_venda'])
# plt.axline(xy1 = (66, 250000), xy2 = (190, 1800000), color='red')

# plt.title('Relação entre preço e área')
# plt.xlabel('Área do primeiro andar (m²)')
# plt.ylabel('Preço de venda (R$)')

# # Qual a reta que melhor se adequa a relação?
# fig = px.scatter(u.dados, x='area_primeiro_andar', y='preco_de_venda', trendline_color_override='red', trendline='ols')
# fig.show()

# Definido x e y
y = u.dados['preco_de_venda']
x = u.dados.drop(columns='preco_de_venda')

# Aplican do o split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 230)

# Dados de treino para usar a fórmula
df_train = pd.DataFrame(data = x_train)
df_train['preco_de_venda'] = y_train

# Ajustando o primeiro modelo
modelo_0 = ols('preco_de_venda ~ area_primeiro_andar', data=df_train).fit()

# Visualizando os parametros
# print(modelo_0.params)

# Resumo do modelo
# print(modelo_0.summary())

# Observando o R²
# print(modelo_0.rsquared)

# Quem são os residuos
# print(modelo_0.resid)
# modelo_0.resid.hist()
# plt.title('Distribuição dos resíduos')
# plt.show()

# Definindo o Y previsto
# y_predict = modelo_0.predict(x_test)

# # Printando o r²
# print(f"R²: {r2_score(y_test, y_predict)}")

# Outras características
sns.pairplot(u.dados, y_vars='preco_de_venda',
             x_vars=['quantidade_banheiros', 'area_segundo_andar','capacidade_carros_garagem'])
plt.show()