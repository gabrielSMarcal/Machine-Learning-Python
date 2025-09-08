import joblib
from sklearn.preprocessing import MinMaxScaler
import utils as u

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

dados = u.dados

# Aplicando K-Means
# mod_kmeans = KMeans(n_clusters=2, random_state=45)
# modelo = mod_kmeans.fit(dados)

# Avaliando o K-Means
# print(mod_kmeans.inertia_)

# print(silhouette_score(dados, mod_kmeans.predict(dados)))

# silhueta, inercia = u.avaliacao(dados)
# print(silhueta)

# Aplicação da avaliação de silhueta
# u.graf_silhueta(2, dados)

# Aplicação do gráfico em cotovelo
# u.plot_cotovelo(inercia) # Teste indica valores altos de inércia, também, cotovelo não é claro

# Aplicando normalização
scaler = MinMaxScaler()
dados_escalados = scaler.fit_transform(dados)

dados_escalados = pd.DataFrame(dados_escalados, columns=dados.columns)
# print(dados_escalados.describe()) # Todos dados tem valores entre 0 e 1

# joblib.dump(scaler, 'scaler.pkl')

# Verificando as métricas para os novos dados
# silhueta, inercia = u.avaliacao(dados_escalados)

# u.graf_silhueta(3, dados_escalados) # Melhor valor de silhueta média

# u.plot_cotovelo(inercia) # Cotovelo mais claro, com valor de cluster melhor em 3

# Criando melhor modelo
modelo_kmeans = KMeans(n_clusters=3, random_state=45, n_init='auto')
modelo_kmeans = modelo_kmeans.fit(dados_escalados)

# joblib.dump(modelo_kmeans, 'modelo_kmeans.pkl')

# Revertendo escala
dados_analise = pd.DataFrame()
dados_analise[dados_escalados.columns] = scaler.inverse_transform(dados_escalados)

dados_analise['cluster'] = modelo_kmeans.labels_

# Agrupando por tipo de cluster
cluster_media = dados_analise.groupby('cluster').mean()

cluster_media = cluster_media.transpose()

cluster_media.columns = ['Cluster 0', 'Cluster 1', 'Cluster 2']
print(cluster_media)