import joblib
import utils as u

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

dados = u.dados

# Aplicando K-Means
mod_kmeans = KMeans(n_clusters=2, random_state=45)
modelo = mod_kmeans.fit(dados)

# Avaliando o K-Means
# print(mod_kmeans.inertia_)

# print(silhouette_score(dados, mod_kmeans.predict(dados)))

silhueta, inercia = u.avaliacao(dados)
# print(silhueta)

# Aplicação da avaliação de silhueta
# u.graf_silhueta(2, dados)

# Aplicação do gráfico em cotovelo
u.plot_cotovelo(inercia) # Teste indica valores altos de inércia, também, cotovelo não é claro