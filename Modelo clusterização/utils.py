import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder

# Carregando dados não rotulados
csv = 'dados_mkt.csv'
df = pd.read_csv(csv)

# Sexo possui Dtype object, precisando converter
# print(df.info())

# F, M e NE
# print(df['sexo'].unique()) 

# Aplicando OneHotEncoder
encoder = OneHotEncoder(categories=[['F', 'M', 'NE']], sparse_output=False)
encoded_sexo = encoder.fit_transform(df[['sexo']])
encoded_df = pd.DataFrame(encoded_sexo, columns=encoder.get_feature_names_out(['sexo']))

# Concatenando o DataFrame original com o DataFrame codificado
dados = pd.concat([df, encoded_df], axis=1).drop('sexo', axis=1)
# print(dados.info())

# Criando arquivo para uso posterior
# joblib.dump(encoder, 'encoder.pkl')

def avaliacao(dados):
    """Função para avaliar os dados"""
    inercia = []
    silhueta = []
    
    for k in range(2, 21):
        kmeans = KMeans(n_clusters=k, random_state=45, n_init='auto')
        kmeans.fit(dados)
        inercia.append(kmeans.inertia_)
        silhueta.append(f'k={k}: - ' + str(silhouette_score(dados, kmeans.predict(dados))))
    return silhueta, inercia
        