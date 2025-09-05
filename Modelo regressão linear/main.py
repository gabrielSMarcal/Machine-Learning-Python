import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import pandas as pd
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

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
# sns.pairplot(u.dados, y_vars='preco_de_venda',
#              x_vars=['quantidade_banheiros', 'area_segundo_andar','capacidade_carros_garagem'])
# plt.show()

# Adicionando constantes
x_train = sm.add_constant(x_train)

# Criando o modelo de regressão (sem fórmula): saturado
modelo_1 = sm.OLS(y_train, x_train[['const', 'area_primeiro_andar', 'existe_segundo_andar',
                                    'area_segundo_andar', 'quantidade_banheiros',
                                    'capacidade_carros_garagem',
                                    'qualidade_da_cozinha_Excelente']]).fit()

# Modelo sem a área do segundo andar
modelo_2 = sm.OLS(y_train, x_train[['const', 'area_primeiro_andar', 'existe_segundo_andar',
                                    'quantidade_banheiros','capacidade_carros_garagem',
                                    'qualidade_da_cozinha_Excelente']]).fit()

# Modelo sem informações sobre garagem
modelo_3 = sm.OLS(y_train, x_train[['const', 'area_primeiro_andar', 'existe_segundo_andar',
                                    'quantidade_banheiros','qualidade_da_cozinha_Excelente']]).fit()

# Resumo do modelo 1
# print(modelo_1.summary())

# Resumo do modelo 2
# print(modelo_2.summary())

# Resumo do modelo 3
# print(modelo_3.summary())

# Comparação dos modelos
# print('R²')
# print("modelo 0:", modelo_0.rsquared)
# print("modelo 1:", modelo_1.rsquared)
# print("modelo 2:", modelo_2.rsquared)
# print("modelo 3:", modelo_3.rsquared)

# # Quantos parametros estão no modelo?
# print("modelo 0:", len(modelo_0.params))
# print("modelo 1:", len(modelo_1.params))
# print("modelo 2:", len(modelo_2.params))
# print("modelo 3:", len(modelo_3.params))

# Constante em x_test
x_test = sm.add_constant(x_test)

# Prevendo com o modelo 3
predict_3 = modelo_3.predict(x_test[['const', 'area_primeiro_andar', 'existe_segundo_andar',
                         'quantidade_banheiros','qualidade_da_cozinha_Excelente']])

# Qual o R² da previsão?
# print(modelo_3.rsquared)

# Qual o R² do treino?
# print(f"R² do teste: {r2_score(y_test, predict_3) * 100:.2f}%")

# # Novo imovel para teste
# novo_imovel = pd.DataFrame({
#     'const': [1],
#     'area_primeiro_andar': [120],
#     'existe_segundo_andar': [1],
#     'quantidade_banheiros': [2],
#     'qualidade_da_cozinha_Excelente': [0]
# })

# # Qual o preço desse imóvel com o modelo 0?
# preco_0 = modelo_0.predict(novo_imovel[['area_primeiro_andar']])
# print(f"Preço do imóvel pelo modelo 0: R$ {preco_0.values[0]:.2f}")

# # Qual o preço desse imóvel com o modelo 3?
# preco_3 = modelo_3.predict(novo_imovel)
# print(f"Preço do imóvel pelo modelo 3: R$ {preco_3.values[0]:.2f}")

# Adicionando constante
# u.d_novo = sm.add_constant(u.d_novo)

# # Qual o preço dos novos imóveis?
# preco_novo = modelo_3.predict(u.d_novo)
# for i, preco in enumerate(preco_novo):
#     print(f"Preço do imóvel {i+1} pelo modelo 3: R$ {preco:.2f}")

# Salvando modelo em um arquivo
# nome_arquivo = 'modelo_regressao_linear.pkl'

# with open(nome_arquivo, 'wb') as arquivo:
#    pickle.dump(modelo_3, arquivo)
   
# # Carregar o modelo de volta do arquivo
# with open(nome_arquivo, 'rb') as arquivo:
#     modelo_carregado = pickle.load(arquivo)

explicativas_1 = ['const','area_primeiro_andar', 'existe_segundo_andar',
       'area_segundo_andar', 'quantidade_banheiros',
       'capacidade_carros_garagem', 'qualidade_da_cozinha_Excelente']
             
explicativas_2 = ['const','area_primeiro_andar', 'existe_segundo_andar',
       'quantidade_banheiros', 'capacidade_carros_garagem',
       'qualidade_da_cozinha_Excelente']

explicativas_3 = ['const','area_primeiro_andar', 'existe_segundo_andar',
       'quantidade_banheiros', 'qualidade_da_cozinha_Excelente']

# VIF 1
vif_1 = pd.DataFrame()
vif_1['variavel'] = explicativas_1

vif_1['vif'] = [variance_inflation_factor(x_train[explicativas_1], i) for i in range(len(explicativas_1))]
# print(vif_1)

# VIF 3
vif_3 = pd.DataFrame()
vif_3['variavel'] = explicativas_3
vif_3['vif'] = [variance_inflation_factor(x_train[explicativas_3], i) for i in range(len(explicativas_3))]
print(vif_3)
