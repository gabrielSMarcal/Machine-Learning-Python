import pandas as pd

dados = pd.read_csv('Preços_de_casas.csv')

dados = dados.drop(columns='Id')

# Correlação
corr = dados.corr()


