# Leitura dos dados
import pandas as pd

# Atividade 1
import seaborn as sns
import matplotlib.pyplot as plt

# Atividade 2
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


dados = pd.read_csv('hoteis.csv')
print(dados.head())

# 1. análise inicial com o PairPlot da Seaborn;

sns.pairplot(dados)
plt.show()

# 2. construir modelos de regressão linear;

X = dados[['Estrelas', 'ProximidadeTurismo', 'Capacidade']]
y = dados['Preco']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

modelo_0 = sm.OLS(y_train, sm.add_constant(x_train)).fit()
print(modelo_0.summary())

# 3. realizar a comparação desses modelos.


# Modelo 1: sem a variável ProximidadeTurismo
x1 = x_train[['Estrelas', 'Capacidade']]
x1 = sm.add_constant(x1)
modelo_1 = sm.OLS(y_train, x1).fit()

# Modelo 2: sem a variável Capacidade
x2 = x_train[['Estrelas', 'ProximidadeTurismo']]
x2 = sm.add_constant(x2)
modelo_2 = sm.OLS(y_train, x2).fit()

# Modelo 3: apenas com a variável Estrelas
x3 = x_train[['Estrelas']]
x3 = sm.add_constant(x3)
modelo_3 = sm.OLS(y_train, x3).fit()

# Comparação dos modelos
print("Modelo 0:", modelo_0.summary()) # 92% de R², porém ProximidadeTurismo com p-valor
print("Modelo 1:", modelo_1.summary()) # 91% de R²
print("Modelo 2:", modelo_2.summary()) # 91% de R², porém ProximidadeTurismo com p-valor
print("Modelo 3:", modelo_3.summary()) # 90% de R²