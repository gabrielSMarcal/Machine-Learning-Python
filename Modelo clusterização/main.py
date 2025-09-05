import utils as u
from sklearn.cluster import KMeans

dados = u.dados

# Aplicando K-Means
mod_kmeans = KMeans(n_clusters=2, random_state=45)
modelo = mod_kmeans.fit(dados)