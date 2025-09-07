import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples

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
    inercia = []
    silhueta = []
    
    for k in range(2, 21):
        kmeans = KMeans(n_clusters=k, random_state=45, n_init='auto')
        kmeans.fit(dados)
        inercia.append(kmeans.inertia_)
        silhueta.append(f'k={k}: - ' + str(silhouette_score(dados, kmeans.predict(dados))))
    return silhueta, inercia

def graf_silhueta (n_clusters, dados):
    
    # Aplica o KMeans ao conjunto de dados
    kmeans = KMeans(n_clusters=n_clusters, random_state=45, n_init='auto')
    cluster_previsoes = kmeans.fit_predict(dados)
    
    # Calcula o silhouette score médio
    silhueta_media = silhouette_score(dados, cluster_previsoes)
    print(f'Valor médio para {n_clusters} clusters: {silhueta_media:.3f}')
    
    # Calcula a pontuação da silhueta para cada amostra
    silhueta_amostra = silhouette_samples(dados, cluster_previsoes)
    
    # Configuração da futra para o grádico de silhueta
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(9, 7)
    
    # Limites do gráfico de silhueta
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(dados) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhueta_amostra = silhueta_amostra[cluster_previsoes == i]
        ith_cluster_silhueta_amostra.sort()
        
        tamanho_cluster_i = ith_cluster_silhueta_amostra.shape[0]
        y_upper = y_lower + tamanho_cluster_i
        
        cor = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhueta_amostra,
                          facecolor=cor, edgecolor=cor, alpha=0.7)
        
        ax1.text(-0.05, y_lower + 0.5 * tamanho_cluster_i, str(i))
        y_lower = y_upper + 10  # Esspaço entre os gráficos
        
    # Linha vertical para a média do Silhouette Score
    ax1.axvline(x=silhueta_media, color='red', linestyle='--')
    
    ax1.set_title(f'Gráfico da Silhueta para {n_clusters} clusters')
    ax1.set_xlabel('Valores do coeficiente da silhueta')
    ax1.set_ylabel('Rótulo do cluster')
    
    ax1.set_yticks([]) # Remove os ticks do eixo y
    ax1.set_xticks([i / 10.0 for i in range (-1, 11)])
    
    plt.show()