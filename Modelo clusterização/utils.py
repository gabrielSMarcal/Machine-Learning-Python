import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder

# Carregando dados não rotulados
csv = 'dados_mkt.csv'
df = pd.read_csv(csv)

# Sexo possui Dtype object, precisando converter
# print(df.info())

# F, M e NE
# print(df['sexo'].unique()) 

# Aplicando OneHotEncoder
enconder = OneHotEncoder(categories=[['F', 'M', 'NE']], sparse_output=False)